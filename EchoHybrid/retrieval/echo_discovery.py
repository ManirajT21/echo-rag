import re
from collections import Counter
from typing import List, Dict, Set
import numpy as np

class EchoDiscovery:
    def __init__(self):
        self.rare_token_threshold = 3
        self.number_repetition_threshold = 2
        self.entity_repetition_threshold = 2

    def _extract_rare_tokens(self, texts: List[str], token_counter: Dict[str, int]) -> List[str]:
        """Extract tokens that appear less than threshold times in the corpus."""
        all_tokens = []
        for text in texts:
            if not text:
                continue
            tokens = re.findall(r'\b\w{4,}\b', text.lower())
            rare_tokens = [t for t in tokens if token_counter.get(t, 0) < self.rare_token_threshold]
            all_tokens.extend(rare_tokens)
        
        # Get most common rare tokens
        counter = Counter(all_tokens)
        return [token for token, count in counter.most_common(10)]

    def _extract_repeated_numbers(self, texts: List[str]) -> Set[str]:
        """Extract numbers that appear in multiple texts."""
        number_counter = Counter()
        for text in texts:
            if not text:
                continue
            numbers = re.findall(r'\b\d+[\d,.]*\b', text)
            number_counter.update(set(numbers))  # Count unique numbers per document
        
        return {num for num, count in number_counter.items() 
                if count >= self.number_repetition_threshold}

    def _extract_repeated_entities(self, texts: List[str]) -> Set[str]:
        """Extract named entities that appear in multiple texts."""
        entity_counter = Counter()
        for text in texts:
            if not text:
                continue
            # Simple heuristic for named entities: consecutive capitalized words
            entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
            entity_counter.update(set(entities))
        
        return {ent for ent, count in entity_counter.items() 
                if count >= self.entity_repetition_threshold}

    def _generate_echo_queries(self, 
                             rare_tokens: List[str],
                             repeated_numbers: Set[str],
                             repeated_entities: Set[str]) -> List[str]:
        """Generate echo queries from discovered patterns."""
        queries = []
        
        # 1. Top rare tokens query
        if rare_tokens:
            queries.append(' '.join(rare_tokens[:5]))
        
        # 2. Number-focused query
        if repeated_numbers:
            number_query = ' '.join(list(repeated_numbers)[:3])
            if rare_tokens:  # Combine with a rare token if available
                number_query += ' ' + rare_tokens[0]
            queries.append(number_query)
        
        # 3. Entity-focused query
        if repeated_entities:
            entity_query = ' '.join(list(repeated_entities)[:2])
            queries.append(entity_query)
        
        # 4. Hybrid query (numbers + entities)
        if repeated_numbers and repeated_entities:
            hybrid_parts = []
            if repeated_numbers:
                hybrid_parts.extend(list(repeated_numbers)[:2])
            if repeated_entities:
                hybrid_parts.extend(list(repeated_entities)[:2])
            queries.append(' '.join(hybrid_parts))
        
        # 5. Longest rare token sequence
        if rare_tokens:
            # Find the longest sequence of rare tokens that appeared together in any document
            max_sequence = []
            current_sequence = []
            
            for token in rare_tokens:
                if not current_sequence:
                    current_sequence = [token]
                else:
                    # Check if this token appeared after the last token in any document
                    current_sequence.append(token)
                
                if len(current_sequence) > len(max_sequence):
                    max_sequence = current_sequence.copy()
            
            if len(max_sequence) >= 2:  # Only add if we found a sequence
                queries.append(' '.join(max_sequence))
        
        # Ensure we return exactly 5 unique queries, padding with empty strings if needed
        queries = list(set(queries))[:5]  # Remove duplicates and limit to 5
        queries.extend([''] * (5 - len(queries)))  # Pad with empty strings if needed
        
        return queries[:5]  # Return exactly 5 queries

    def discover_echo_queries(self, search_results: Dict[str, List[Dict]], token_counter: Dict[str, int]) -> List[str]:
        """
        Generate echo queries from search results.
        
        Args:
            search_results: Dictionary containing 'dense', 'sparse', and 'temporal' search results
            token_counter: Dictionary of token frequencies in the corpus
            
        Returns:
            List of 5 echo queries (may include empty strings if not enough patterns found)
        """
        # Extract text from all results
        all_texts = []
        for result_type in ['dense', 'sparse', 'temporal']:
            for result in search_results.get(result_type, []):
                if 'payload' in result and 'text' in result['payload']:
                    all_texts.append(result['payload']['text'])
                elif 'text' in result:
                    all_texts.append(result['text'])
        
        # Extract patterns
        rare_tokens = self._extract_rare_tokens(all_texts, token_counter)
        repeated_numbers = self._extract_repeated_numbers(all_texts)
        repeated_entities = self._extract_repeated_entities(all_texts)
        
        # Generate echo queries
        return self._generate_echo_queries(rare_tokens, repeated_numbers, repeated_entities)
