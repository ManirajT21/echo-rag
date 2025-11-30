import re
from typing import Tuple
import numpy as np
from collections import Counter

class RecipeSelector:
    @staticmethod
    def _has_numbers(text: str) -> bool:
        return bool(re.search(r'\d', text))
    
    @staticmethod
    def _count_rare_tokens(text: str, token_counter: Counter) -> int:
        tokens = re.findall(r'\b\w{4,}\b', text.lower())
        return sum(1 for token in tokens if token_counter.get(token, 0) < 3)
    
    @staticmethod
    def _count_named_entities(text: str) -> int:
        # Simple heuristic for named entities: consecutive capitalized words
        return len(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))
    
    @staticmethod
    def calculate_weights(query: str, token_counter: Counter) -> Tuple[float, float, float]:
        """
        Calculate weights for dense, sparse, and temporal components.
        Returns: (w_dense, w_sparse, w_temporal)
        """
        q = query.strip()
        has_numbers = RecipeSelector._has_numbers(q)
        rare_tokens = RecipeSelector._count_rare_tokens(q, token_counter)
        named_entities = RecipeSelector._count_named_entities(q)

        # Base weights (slightly favor dense by default for conceptual questions)
        w_dense = 0.5
        w_sparse = 0.35
        w_temporal = 0.15

        # If query is clearly about definition/explanation, push more into dense
        if re.match(r"(?i)^(what is|define|explain)\b", q):
            w_dense += 0.15
            w_sparse -= 0.10
            w_temporal -= 0.05

        # Numbers → more temporal, slightly less dense/sparse
        if has_numbers:
            w_temporal += 0.20
            w_dense -= 0.10
            w_sparse -= 0.10

        # Many rare tokens → more sparse emphasis
        if rare_tokens > 2:
            w_sparse += 0.20
            w_dense -= 0.10
            w_temporal -= 0.10

        # Multiple named entities → dense + sparse, less temporal
        if named_entities > 1:
            w_dense += 0.10
            w_sparse += 0.10
            w_temporal -= 0.20

        # Clamp so nothing goes negative or too tiny
        w_dense = max(w_dense, 0.05)
        w_sparse = max(w_sparse, 0.05)
        w_temporal = max(w_temporal, 0.05)

        total = w_dense + w_sparse + w_temporal
        return (w_dense / total, w_sparse / total, w_temporal / total)
