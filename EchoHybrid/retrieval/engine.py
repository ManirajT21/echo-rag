import asyncio
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from .recipe_selector import RecipeSelector
from .triple_vector_search import TripleVectorSearch
from .echo_discovery import EchoDiscovery
from .echo_parallel_search import EchoParallelSearch
from .rrf_fusion import RRFFusion
from .echo_parent import EchoParent
from .negative_memory import NegativeMemory
from .confidence_gate import ConfidenceGate

# ==================================================
# Global Chunk Index
# ==================================================
BASE_DIR = Path(__file__).resolve().parent.parent
CHUNK_DIR = BASE_DIR / "chunking_output"

def build_chunk_index() -> Dict[tuple, Dict[str, Any]]:
    """Build an index of all chunks for fast lookup."""
    index = {}
    for path in CHUNK_DIR.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            for c in chunks:
                source = c.get("metadata", {}).get("source_file") or c.get("source_file")
                cid = c.get("chunk_id")
                if not source or not cid:
                    continue
                text = (
                    c.get("text")
                    or c.get("content")
                    or next((v for v in c.values() if isinstance(v, str)), "")
                )
                index[(source, cid)] = {
                    "text": text,
                    "metadata": c.get("metadata", {})
                }
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not process {path}: {str(e)}")
            continue
    return index

# Build the chunk index at module load time
CHUNK_INDEX = build_chunk_index()

def hydrate_from_chunks(payload: Dict[str, Any]) -> tuple[str, str]:
    """
    Retrieve the actual text content from the chunk index.
    
    Args:
        payload: The payload from Qdrant search result
        
    Returns:
        tuple: (text, page_or_time)
    """
    source = payload.get("source") or payload.get("source_file")
    chunk_id = payload.get("chunk_id")

    text = ""
    page_or_time = ""

    if source and chunk_id:
        info = CHUNK_INDEX.get((source, chunk_id))
        if info:
            text = info.get("text", "")
            meta = info.get("metadata", {})
            page_or_time = meta.get("page_no") or meta.get("page_label") or ""

    return text, str(page_or_time) if page_or_time is not None else ""

class EchoHybridEngine:
    def __init__(self, qdrant_url: str = "http://localhost:6333", model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the EchoHybrid retrieval engine.
        
        Args:
            qdrant_url: URL of the Qdrant server
            model_name: Name of the SentenceTransformer model to use
        """
        self.qdrant = QdrantClient(url=qdrant_url)
        self.model = SentenceTransformer(model_name)
        
        # Initialize components
        self.recipe_selector = RecipeSelector()
        self.triple_search = TripleVectorSearch(self.qdrant, self.model)
        self.echo_discovery = EchoDiscovery()
        self.echo_searcher = EchoParallelSearch(self.qdrant, self.model, self.triple_search.token_counter)
        self.rrf_fusion = RRFFusion()
        self.echo_parent = EchoParent()
        self.negative_memory = NegativeMemory(self.model)
        self.confidence_gate = ConfidenceGate()
    
    async def _format_results(self, results: List[Dict]) -> List[Dict]:
        """Format results into the required output format with hydrated text from chunks."""
        formatted = []
        for i, result in enumerate(results, 1):
            payload = result.get('payload', {})
            
            # Hydrate the text from chunk files
            text, page_or_time = hydrate_from_chunks(payload)
            
            formatted.append({
                'rank': i,
                'score': round(float(result.get('score', 0)), 3),
                'text': text,
                'source_file': payload.get('source_file') or payload.get('source', ''),
                'page_or_time': page_or_time,
                'highlight_terms': result.get('matched_terms', []),
                'found_in_round': 'initial' if i == 0 else f'echo_{i}',
                'modality': 'text'
            })
        return formatted
    
    async def run_echohybrid_query(self, user_query: str) -> List[Dict]:
        """
        Run the full EchoHybrid retrieval pipeline.
        
        Args:
            user_query: The user's search query
            
        Returns:
            List of results in the required format
        """
        try:
            # Stage 1: Dynamic Recipe Weights
            token_counter = self.triple_search.token_counter
            w_dense, w_sparse, w_temporal = self.recipe_selector.calculate_weights(user_query, token_counter)
            
            # Stage 2: Triple Vector Search
            search_results = self.triple_search.search(user_query, top_k=40)
            
            # Stage 3: Echo Discovery
            echo_queries = self.echo_discovery.discover_echo_queries(search_results, token_counter)
            
            # Stage 4: Parallel Echo Searches
            echo_results = await self.echo_searcher.search(echo_queries, top_k=30)
            
            # Prepare results for RRF fusion
            all_results = [
                {'round_name': 'initial', 'results': search_results.get('dense', [])},
                *[{'round_name': name, 'results': results} 
                  for name, results in echo_results.items() if results]
            ]
            
            # Stage 5: Reciprocal Rank Fusion
            fused_results = self.rrf_fusion.fuse(all_results)
            
            # If no results found, return early
            if not fused_results:
                return [{
                    'rank': 1,
                    'score': 0.0,
                    'text': "No relevant information found for your query.",
                    'source_file': "",
                    'page_or_time': "",
                    'highlight_terms': [],
                    'found_in_round': "no_results",
                    'modality': "text"
                }]
            
            # Stage 6: EchoParent Boost
            boosted_results = self.echo_parent.apply_parent_boost(fused_results)
            
            # Stage 7: Negative Memory Penalties
            query_embedding = self.model.encode(user_query)
            final_results = self.negative_memory.apply_penalties(query_embedding, boosted_results)
            
            # Stage 8: Confidence Gate
            confidence_result = self.confidence_gate.filter_results(final_results)
            
            # Format and return results
            if confidence_result.get('is_high_confidence', False):
                formatted_results = await self._format_results(confidence_result.get('results', []))
                if not formatted_results:
                    return [{
                        'rank': 1,
                        'score': 0.0,
                        'text': "No relevant information found for your query.",
                        'source_file': "",
                        'page_or_time': "",
                        'highlight_terms': [],
                        'found_in_round': "no_results",
                        'modality': "text"
                    }]
                return formatted_results
            else:
                # Return low-confidence response
                return [{
                    'rank': 1,
                    'score': confidence_result.get('max_score', 0.0),
                    'text': "I don't have enough confidence in the search results to provide a reliable answer.",
                    'source_file': "",
                    'page_or_time': "",
                    'highlight_terms': [],
                    'found_in_round': "low_confidence",
                    'modality': "text"
                }]
                
        except Exception as e:
            print(f"Error in run_echohybrid_query: {str(e)}")
            # Return error response
            return [{
                'rank': 1,
                'score': 0.0,
                'text': f"An error occurred while processing your query: {str(e)}",
                'source_file': "",
                'page_or_time': "",
                'highlight_terms': [],
                'found_in_round': result.get('found_in_round', 'initial'),
                'modality': 'text'
            }]
            
def main():
    """CLI entry point for the EchoHybrid retrieval engine."""
    import argparse
    import asyncio
    
    parser = argparse.ArgumentParser(description='EchoHybrid Retrieval Engine')
    parser.add_argument('query', type=str, help='Search query')
    parser.add_argument('--qdrant', type=str, default="http://localhost:6333",
                       help='Qdrant server URL')
    parser.add_argument('--model', type=str, default="all-MiniLM-L6-v2",
                       help='SentenceTransformer model name')
    
    args = parser.parse_args()
    
    # Initialize and run the engine
    engine = EchoHybridEngine(qdrant_url=args.qdrant, model_name=args.model)
    
    async def run_query():
        results = await engine.run_echohybrid_query(args.query)
        print(json.dumps(results, indent=2, ensure_ascii=False))
    
    asyncio.run(run_query())

if __name__ == "__main__":
    main()
