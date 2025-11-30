from typing import List, Dict, Tuple
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import json
import os
import re
from pathlib import Path

class TripleVectorSearch:
    def __init__(self, qdrant_client: QdrantClient, model: SentenceTransformer):
        self.qdrant = qdrant_client
        self.model = model
        self.collection_name = "document_embeddings"
        self.chunk_dir = Path("chunking_output")
        self.token_counter = self._build_token_counter()
    
    def _build_token_counter(self) -> dict:
        """Build a token frequency counter from all chunks."""
        counter = defaultdict(int)
        for chunk_file in self.chunk_dir.glob("*.json"):
            with open(chunk_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if 'text' in item:
                            tokens = re.findall(r'\b\w{4,}\b', item['text'].lower())
                            for token in tokens:
                                counter[token] += 1
        return counter
    
    def _get_sparse_embedding(self, text: str) -> Dict[int, float]:
        """Simple TF-IDF like sparse embedding."""
        tokens = re.findall(r'\b\w{4,}\b', text.lower())
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        
        # Apply IDF weighting
        sparse_vec = {}
        for token, count in tf.items():
            doc_freq = self.token_counter.get(token, 1)
            idf = np.log(len(self.token_counter) / (1 + doc_freq))
            sparse_vec[hash(token) % (2**32)] = count * idf
            
        return sparse_vec
    
    def _temporal_score(self, query: str, chunk: dict) -> float:
        """Score based on temporal/numeric patterns."""
        score = 0.0
        query_nums = set(re.findall(r'\d+', query))
        
        if 'text' in chunk:
            chunk_nums = set(re.findall(r'\d+', chunk['text']))
            matching_nums = query_nums.intersection(chunk_nums)
            if matching_nums:
                score += 0.5 * len(matching_nums) / len(query_nums)
                
        # Add date pattern matching
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # MM/DD/YYYY or DD-MM-YYYY
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',    # YYYY-MM-DD
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'  # Month Day, Year
        ]
        
        for pattern in date_patterns:
            query_dates = set(re.findall(pattern, query, re.IGNORECASE))
            if 'text' in chunk:
                chunk_dates = set(re.findall(pattern, chunk['text'], re.IGNORECASE))
                if query_dates and chunk_dates:
                    score += 0.3 * len(query_dates.intersection(chunk_dates)) / len(query_dates)
                    
        return min(1.0, score)  # Cap at 1.0
    
    def search(self, query: str, top_k: int = 40) -> Dict[str, List[Dict]]:
        """Perform triple vector search and return combined results.
        
        Args:
            query: The search query string
            top_k: Number of results to return for each search type
            
        Returns:
            Dictionary containing 'dense', 'sparse', and 'temporal' search results
        """
        # Dense search using vector similarity
        query_embedding = self.model.encode(query).tolist()
        
        try:
            # Get the collection info to find the vector configuration
            collection_info = self.qdrant.get_collection(self.collection_name)
            
            # Use query_points with the correct parameters
            response = self.qdrant.query_points(
                collection_name=self.collection_name,
                query=query_embedding,  # The query vector
                limit=top_k,
                with_vectors=False,
                with_payload=True
            )
            
            # Convert search results to the expected format
            dense_results = []
            if hasattr(response, 'points') and response.points:
                for point in response.points:
                    dense_results.append({
                        'id': str(point.id) if hasattr(point, 'id') else '',
                        'score': float(point.score) if hasattr(point, 'score') else 0.0,
                        'payload': point.payload if hasattr(point, 'payload') else {}
                    })
            else:
                print(f"No points found in collection {self.collection_name}")
            
        except Exception as e:
            print(f"Error in dense search: {str(e)}")
            dense_results = []
        
        # For now, use the same results for sparse search
        # In a production system, you'd implement BM25 or other sparse retrieval here
        sparse_results = dense_results.copy() if dense_results else []
        
        # Temporal search (boost chunks with matching numbers/dates)
        temporal_results = []
        try:
            all_chunks = []
            for chunk_file in self.chunk_dir.glob("*.json"):
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_chunks.extend(data)
            
            # Score chunks based on temporal patterns
            scored_chunks = []
            for chunk in all_chunks:
                if not isinstance(chunk, dict):
                    continue
                score = self._temporal_score(query, chunk)
                if score > 0:
                    scored_chunks.append({
                        'id': chunk.get('id', str(hash(str(chunk)))),
                        'score': score,
                        'payload': chunk
                    })
            
            # Sort and get top-k temporal results
            temporal_results = sorted(scored_chunks, key=lambda x: x['score'], reverse=True)[:top_k]
            
        except Exception as e:
            print(f"Error in temporal search: {str(e)}")
            temporal_results = []
        
        return {
            'dense': dense_results,
            'sparse': sparse_results,
            'temporal': temporal_results
        }
        scored_chunks = []
        for chunk in all_chunks:
            score = self._temporal_score(query, chunk)
            if score > 0:
                scored_chunks.append({
                    'id': chunk.get('id', str(hash(str(chunk)))),
                    'score': score,
                    'payload': chunk
                })
        
        # Sort and get top-k temporal results
        temporal_results = sorted(scored_chunks, key=lambda x: x['score'], reverse=True)[:top_k]
        
        return {
            'dense': dense_results,
            'sparse': sparse_results,
            'temporal': temporal_results
        }
