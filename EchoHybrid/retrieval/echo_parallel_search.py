import asyncio
import re
from typing import List, Dict, Any
from qdrant_client import QdrantClient
import numpy as np
import json
from pathlib import Path

class EchoParallelSearch:
    def __init__(self, qdrant_client: QdrantClient, model, token_counter: dict):
        self.qdrant = qdrant_client
        self.model = model
        self.token_counter = token_counter
        self.chunk_dir = Path("chunking_output")
        
    async def _sparse_search(self, query: str, top_k: int = 30) -> List[Dict]:
        """Perform sparse search using TF-IDF like scoring."""
        tokens = [t for t in re.findall(r'\b\w{4,}\b', query.lower()) 
                 if self.token_counter.get(t, 0) < 10]  # Focus on rarer terms
        
        if not tokens:
            return []
            
        # Simple BM25-like scoring
        results = []
        for chunk_file in self.chunk_dir.glob("*.json"):
            with open(chunk_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
                for chunk in chunks:
                    if not isinstance(chunk, dict) or 'text' not in chunk:
                        continue
                        
                    chunk_text = chunk['text'].lower()
                    score = 0
                    matched_terms = []
                    
                    for token in tokens:
                        count = chunk_text.count(token)
                        if count > 0:
                            idf = np.log(len(self.token_counter) / (1 + self.token_counter.get(token, 1)))
                            score += (count * (1.2 + 1) / (count + 1.2 * (1 - 0.75 + 0.75 * len(chunk_text) / 100))) * idf
                            matched_terms.append(token)
                    
                    if score > 0:
                        result = chunk.copy()
                        result['score'] = score
                        result['matched_terms'] = matched_terms
                        results.append(result)
        
        # Sort by score and return top-k
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return results[:top_k]

    async def _batch_search(self, queries: List[str], top_k: int = 30) -> Dict[str, List[Dict]]:
        """Run multiple sparse searches in parallel."""
        tasks = [self._sparse_search(query, top_k) for query in queries if query.strip()]
        results = await asyncio.gather(*tasks)
        return {f"echo_{i+1}": result for i, result in enumerate(results)}

    async def search(self, echo_queries: List[str], top_k: int = 30) -> Dict[str, List[Dict]]:
        """
        Execute all echo queries in parallel.
        
        Args:
            echo_queries: List of up to 5 echo queries
            top_k: Number of results to return per query
            
        Returns:
            Dictionary mapping query names to their results
        """
        if not echo_queries:
            return {}
            
        return await self._batch_search(echo_queries, top_k)
