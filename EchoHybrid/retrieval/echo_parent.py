from typing import List, Dict, Any
from collections import defaultdict
import json
from pathlib import Path

class EchoParent:
    def __init__(self, boost_factor: float = 0.38):
        self.boost_factor = boost_factor
        self.parent_map = self._build_parent_map()
    
    def _build_parent_map(self) -> Dict[str, Dict]:
        """Build a map of chunk IDs to their parent information."""
        parent_map = {}
        chunk_dir = Path("chunking_output")
        
        for chunk_file in chunk_dir.glob("*.json"):
            with open(chunk_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
                for chunk in chunks:
                    if not isinstance(chunk, dict) or 'id' not in chunk:
                        continue
                    
                    # Extract parent chain if available, otherwise use file as parent
                    parent_chain = chunk.get('parent_chain', [
                        {
                            'type': 'file',
                            'id': chunk.get('source_file', str(chunk_file)),
                            'text': chunk.get('text', '')[:100] + '...'
                        }
                    ])
                    
                    parent_map[chunk['id']] = {
                        'parent_chain': parent_chain,
                        'text': chunk.get('text', '')[:200] + '...' if chunk.get('text') else ''
                    }
        
        return parent_map
    
    def _get_parent_chain(self, chunk_id: str) -> List[Dict]:
        """Get the parent chain for a chunk."""
        return self.parent_map.get(chunk_id, {}).get('parent_chain', [])
    
    def apply_parent_boost(self, results: List[Dict], top_k: int = 20) -> List[Dict]:
        """
        Apply parent-based score boosting to top-k results.
        
        Args:
            results: List of search results with scores
            top_k: Number of top results to consider for boosting
            
        Returns:
            List of results with boosted scores where applicable
        """
        if not results:
            return []
            
        # Get top-k results for parent analysis
        top_results = results[:top_k]
        
        # Count parent occurrences
        parent_counter = defaultdict(int)
        for result in top_results:
            chunk_id = result.get('id')
            parent_chain = self._get_parent_chain(chunk_id)
            for parent in parent_chain:
                parent_key = f"{parent.get('type')}:{parent.get('id')}"
                parent_counter[parent_key] += 1
        
        # Apply boosts based on parent frequency
        boosted_results = []
        for result in results:
            result = result.copy()  # Don't modify original
            chunk_id = result.get('id')
            parent_chain = self._get_parent_chain(chunk_id)
            
            # Calculate boost based on parent frequency
            boost = 0.0
            for parent in parent_chain:
                parent_key = f"{parent.get('type')}:{parent.get('id')}"
                if parent_counter.get(parent_key, 0) > 1:  # Only boost if parent appears multiple times
                    boost += self.boost_factor
            
            # Apply boost
            if 'score' in result:
                result['original_score'] = result['score']
                result['score'] *= (1.0 + boost)
                result['boost'] = boost
            
            boosted_results.append(result)
        
        # Re-sort results after boosting
        boosted_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return boosted_results
