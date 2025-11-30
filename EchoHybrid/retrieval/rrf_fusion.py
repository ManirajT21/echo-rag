from typing import List, Dict, Any
import numpy as np
from collections import defaultdict

class RRFFusion:
    def __init__(self, k: int = 60):
        self.k = k  # Rank offset
    
    @staticmethod
    def _get_doc_key(doc: Dict) -> str:
        """Generate a unique key for a document based on its content and source."""
        text = doc.get('text', '') or doc.get('payload', {}).get('text', '')
        source = doc.get('source_file', '') or doc.get('payload', {}).get('source_file', '')
        return f"{source}:{text[:100]}"
    
    def _score_document(self, rank: int) -> float:
        """Calculate RRF score for a document at given rank (1-based)."""
        return 1.0 / (self.k + rank)
    
    def fuse(self, result_sets: List[Dict[str, List[Dict]]]) -> List[Dict]:
        """
        Fuse multiple ranked result sets using Reciprocal Rank Fusion.
        
        Args:
            result_sets: List of dictionaries, each containing 'results' key with a list of documents
            
        Returns:
            List of documents with fused scores, sorted by score in descending order
        """
        doc_scores = defaultdict(float)
        doc_metadata = {}  # Store the first occurrence of each document
        
        for result_set in result_sets:
            if not result_set or 'results' not in result_set:
                continue
                
            for rank, doc in enumerate(result_set['results'], 1):
                if not doc:
                    continue
                    
                doc_key = self._get_doc_key(doc)
                doc_scores[doc_key] += self._score_document(rank)
                
                # Store document metadata if not already present
                if doc_key not in doc_metadata:
                    doc_metadata[doc_key] = {
                        'doc': doc,
                        'found_in_rounds': set()
                    }
                
                # Track which result set this came from
                doc_metadata[doc_key]['found_in_rounds'].add(result_set.get('round_name', 'initial'))
        
        # Convert to list of documents with fused scores
        fused_results = []
        for doc_key, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
            if doc_key in doc_metadata:
                doc_info = doc_metadata[doc_key]
                doc = doc_info['doc'].copy()
                doc['score'] = score
                doc['found_in_rounds'] = list(doc_info['found_in_rounds'])
                fused_results.append(doc)
        
        return fused_results
