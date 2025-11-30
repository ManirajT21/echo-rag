import numpy as np
from typing import List, Dict, Any, Set
from collections import deque
from sentence_transformers import SentenceTransformer
import os
import json
from pathlib import Path

class NegativeMemory:
    def __init__(self, model: SentenceTransformer, max_size: int = 2000, penalty_factor: float = 0.7):
        self.model = model
        self.max_size = max_size
        self.penalty_factor = penalty_factor
        self.negative_embeddings = []
        self.negative_texts = set()
        self.storage_file = Path("negative_memory.json")
        self._load_negative_memory()
    
    def _load_negative_memory(self) -> None:
        """Load negative memory from disk if it exists."""
        if self.storage_file.exists():
            try:
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.negative_embeddings = [np.array(emb) for emb in data.get('embeddings', [])]
                    self.negative_texts = set(data.get('texts', []))
            except Exception as e:
                print(f"Error loading negative memory: {e}")
    
    def _save_negative_memory(self) -> None:
        """Save negative memory to disk."""
        try:
            data = {
                'embeddings': [emb.tolist() for emb in self.negative_embeddings],
                'texts': list(self.negative_texts)
            }
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving negative memory: {e}")
    
    def add_negative_examples(self, texts: List[str]) -> None:
        """Add negative examples to memory."""
        if not texts:
            return
            
        # Get embeddings for new negative examples
        new_embeddings = self.model.encode(texts, show_progress_bar=False)
        
        # Add to memory
        for text, emb in zip(texts, new_embeddings):
            if text in self.negative_texts:
                continue
                
            self.negative_texts.add(text)
            self.negative_embeddings.append(emb)
        
        # Trim if over max size
        if len(self.negative_embeddings) > self.max_size:
            self.negative_embeddings = self.negative_embeddings[-self.max_size:]
            # Rebuild texts set from remaining embeddings
            self.negative_texts = set()
        
        self._save_negative_memory()
    
    def apply_penalties(self, query_embedding: np.ndarray, results: List[Dict]) -> List[Dict]:
        """
        Apply negative memory penalties to search results.
        
        Args:
            query_embedding: The query embedding vector
            results: List of search results with 'embedding' field
            
        Returns:
            List of results with penalties applied
        """
        if not self.negative_embeddings or not results:
            return results
        
        # Convert query embedding to numpy array if needed
        query_emb = np.array(query_embedding).flatten()
        
        # Calculate penalties for each result
        penalized_results = []
        for result in results:
            result = result.copy()  # Don't modify original
            
            if 'embedding' not in result:
                penalized_results.append(result)
                continue
                
            result_emb = np.array(result['embedding']).flatten()
            
            # Calculate max similarity to any negative example
            max_penalty = 0.0
            for neg_emb in self.negative_embeddings:
                # Cosine similarity
                neg_emb = neg_emb.flatten()
                similarity = np.dot(result_emb, neg_emb) / (
                    np.linalg.norm(result_emb) * np.linalg.norm(neg_emb) + 1e-8
                )
                max_penalty = max(max_penalty, max(0, similarity))  # Only penalize positive similarity
            
            # Apply penalty to score if it exists
            if 'score' in result:
                result['original_score'] = result['score']
                result['score'] -= self.penalty_factor * max_penalty
                result['negative_penalty'] = -self.penalty_factor * max_penalty
            
            penalized_results.append(result)
        
        # Re-sort results after applying penalties
        penalized_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return penalized_results
