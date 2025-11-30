from typing import List, Dict, Tuple, Optional
import numpy as np

class ConfidenceGate:
    def __init__(self, min_confidence: float = 0.15, max_results: int = 12):
        """
        Initialize the confidence gate.
        
        Args:
            min_confidence: Minimum score threshold for results (0.0-1.0)
            max_results: Maximum number of results to return
        """
        self.min_confidence = min_confidence
        self.max_results = max_results
    
    def _calculate_confidence(self, scores: List[float]) -> Tuple[float, float]:
        """
        Calculate confidence metrics from result scores.
        
        Args:
            scores: List of result scores
            
        Returns:
            Tuple of (confidence_score, max_score)
        """
        if not scores:
            return 0.0, 0.0
            
        scores = np.array(scores)
        max_score = float(np.max(scores)) if len(scores) > 0 else 0.0
        
        if len(scores) == 1:
            return max_score, max_score
            
        # Calculate confidence based on score distribution
        if max_score > 0:
            # Normalize scores
            normalized = scores / max_score
            # Confidence is based on the gap between top score and others
            if len(scores) > 1:
                gap = normalized[0] - normalized[1] if len(normalized) > 1 else 1.0
                confidence = max_score * (0.7 + 0.3 * gap)
            else:
                confidence = max_score
        else:
            confidence = 0.0
            
        return min(1.0, confidence), max_score
    
    def filter_results(self, results: List[Dict]) -> Dict:
        """
        Filter results based on confidence threshold.
        
        Args:
            results: List of result dictionaries with 'score' key
            
        Returns:
            Dictionary containing:
            - results: Filtered results (empty if low confidence)
            - confidence: Confidence score (0.0-1.0)
            - is_high_confidence: Whether results meet confidence threshold
        """
        if not results:
            return {
                'results': [],
                'confidence': 0.0,
                'is_high_confidence': False
            }
        
        # Extract scores
        scores = [r.get('score', 0) for r in results]
        confidence, max_score = self._calculate_confidence(scores)
        
        # Check if we meet confidence threshold
        is_high_confidence = (max_score >= self.min_confidence)
        
        # Filter and limit results
        filtered = []
        if results:
            # Sort results by score in descending order
            sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
            
            # Take top results that meet the minimum confidence
            filtered = [r for r in sorted_results if r.get('score', 0) >= self.min_confidence]
            
            # If we have some results but not enough, include lower scoring ones
            if len(filtered) < self.max_results and len(sorted_results) > 0:
                remaining = [r for r in sorted_results if r.get('score', 0) < self.min_confidence]
                filtered.extend(remaining[:self.max_results - len(filtered)])
            
            # Limit to max_results
            filtered = filtered[:self.max_results]
            
            # If we still have no results but had some input, return the top result
            if not filtered and sorted_results:
                filtered = [sorted_results[0]]
        
        return {
            'results': filtered,
            'confidence': confidence,
            'is_high_confidence': is_high_confidence,
            'max_score': max_score
        }
