#!/usr/bin/env python3
"""
Document Processing Pipeline Runner
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from ingestion.parser import DocumentParser, parse_and_save
from ingestion.chunker import HybridChunker, chunk_file
from ingestion.redactor import DocumentRedactor, RedactionRule
from ingestion.embedder import DocumentEmbedder, embed_documents
from ingestion.vector_store import QdrantVectorStore


class DocumentProcessor:
    """
    A class to handle the end-to-end document processing pipeline.
    """
    
    def __init__(
        self,
        output_dir: str = "chunking_output",
        redact_sensitive: bool = True,
        max_chunk_size: int = 1000,
        chunk_overlap: int = 200,
        file_pattern: str = "*",
        embed_model: str = "all-MiniLM-L6-v2",
        device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
        qdrant_collection: str = "document_embeddings",  # Updated collection name
        recreate_collection: bool = False
    ):
        """
        Initialize the document processor.
        """
        self.output_dir = Path(output_dir)
        base_dir = Path(__file__).resolve().parent.parent   # Points to EchoHybrid/
        self.embeddings_dir = base_dir / "generated_embeddings"
        self.redact_sensitive = redact_sensitive
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.file_pattern = file_pattern
        self.embed_model = embed_model
        self.device = device
        
        # Initialize parser, chunker, embedder, and vector store
        self.parser = DocumentParser(redact_sensitive=redact_sensitive)
        self.chunker = HybridChunker(
            max_chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap
        )
        # Initialize vector store first
        self.vector_store = QdrantVectorStore(
            collection_name=qdrant_collection,
            vector_size=384,  # all-MiniLM-L6-v2 uses 384 dimensions
            recreate_collection=recreate_collection
        )
        
        # Initialize embedder with vector store
        self.embedder = DocumentEmbedder(
            model_name=embed_model,
            device=device,
            vector_store=self.vector_store
        )
        
        # Create output directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    def process_file(self, input_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Process a single file through the pipeline.
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        print(f"\nProcessing file: {input_path}")
        
        try:
            # Step 1: Parse the document
            print("  - Parsing document...")
            parsed_doc = self.parser.parse_document(input_path)
            
            # Save the parsed output (for debugging)
            parsed_output_path = self.parser.get_output_path(input_path)
            self.parser.save_to_json(parsed_doc, input_path=input_path)
            print(f"  - Parsed output saved to: {parsed_output_path}")
            
            # Step 2: Chunk the document
            print("  - Chunking document...")
            chunks = self.chunker.chunk_document(parsed_doc)
            
            if not chunks:
                print("  - No chunks generated from the document.")
                return []
            
            # Add source file information to each chunk
            for i, chunk in enumerate(chunks):
                chunk_meta = chunk.get('metadata', {})
                chunk_id = f"{input_path.stem}_chunk_{i+1}"
                chunk_meta.update({
                    'source_file': str(input_path.name),
                    'chunk_id': chunk_id,
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                })
                chunk['chunk_id'] = chunk_id
                chunk['chunk_index'] = i
            
            # Save the chunks
            output_path = self._save_chunks(chunks, input_path)
            print(f"  - Chunks saved to: {output_path}")
            
            # Step 3: Generate embeddings for chunks
            print("  - Generating embeddings...")
            
            # Extract text from chunks
            chunk_texts = []
            for chunk in chunks:
                if 'text' in chunk:
                    chunk_texts.append(chunk['text'])
                elif 'content' in chunk:
                    chunk_texts.append(chunk['content'])
                else:
                    text = next((v for v in chunk.values() if isinstance(v, str)), '')
                    chunk_texts.append(text)
            
            # Prepare documents with text and metadata for embedding
            documents = []
            metadatas_for_qdrant = []  # For Qdrant storage (without text)
            for i, (chunk, text) in enumerate(zip(chunks, chunk_texts)):
                chunk_metadata = chunk.get('metadata', {}) if isinstance(chunk.get('metadata'), dict) else {}
                chunk_metadata.update({
                    'source': str(input_path.name),
                    'chunk_id': chunk.get('chunk_id', f"chunk_{i}"),
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                })
                documents.append({
                    'text': text,
                    'metadata': chunk_metadata
                })
                
                # Create metadata without text for Qdrant
                metadata_without_text = {
                    k: v for k, v in chunk_metadata.items() 
                    if k not in ['text', 'content']
                }
                metadatas_for_qdrant.append(metadata_without_text)
            
            # Generate embeddings
            embedding_results = self.embedder.embed_documents(documents)
            
            # Add embeddings to chunks for saving to file
            for i, chunk in enumerate(chunks):
                if i < len(embedding_results):
                    chunk['embedding'] = embedding_results[i]['embedding']
            
            # Store ONLY EMBEDDINGS in Qdrant
            print("  - Storing embeddings in Qdrant...")
            # Extract embeddings from results for Qdrant
            embeddings_for_qdrant = [result['embedding'] for result in embedding_results]
            self.vector_store.add_embeddings(
                embeddings=embeddings_for_qdrant,
                metadatas=metadatas_for_qdrant  # Metadata without text
            )
            
            # Save chunks with embeddings to file for reference
            output_file = self._save_chunks(chunks, input_path)
            
            print(f"  - Processed {len(chunks)} chunks")
            print(f"  - Chunks saved to {output_file}")
            print(f"  - Embeddings stored in Qdrant collection: {self.vector_store.collection_name}")
            
            return chunks
            
        except Exception as e:
            print(f"Error processing {input_path}: {str(e)}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return []
    
    def process_directory(self, input_dir: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Process all matching files in a directory.
        """
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            raise NotADirectoryError(f"Input is not a directory: {input_dir}")
        
        all_chunks = []
        
        # Find all matching files
        files = list(input_dir.glob(self.file_pattern))
        if not files:
            print(f"No files found matching pattern: {self.file_pattern}")
            return []
        
        print(f"Found {len(files)} files to process...")
        
        # Process each file
        for i, file_path in enumerate(files, 1):
            if file_path.is_file():
                print(f"\nProcessing file {i}/{len(files)}")
                chunks = self.process_file(file_path)
                all_chunks.extend(chunks)
        
        return all_chunks
    
    def _save_chunks(self, chunks: List[Dict[str, Any]], input_path: Path) -> Path:
        """
        Save chunks to a JSON file with timestamp in the filename.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{input_path.stem}_chunks_{timestamp}.json"
        output_path = self.output_dir / output_filename
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the chunks
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        return output_path
        
    def generate_embeddings(
        self, 
        chunks: List[Dict[str, Any]], 
        source_name: str,
        overwrite: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of text chunks.
        """
        print(f"  - Generating embeddings using {self.embed_model}...")
        
        try:
            # Generate embeddings
            results = self.embedder.embed_chunks(chunks)
            
            # Save the embeddings
            output_file = self.embedder.save_embeddings(
                results, 
                source_name=source_name,
                overwrite=overwrite
            )
            
            print(f"  - Embeddings saved to: {output_file}")
            return [r.to_dict() for r in results]
            
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            return []
    
    def process_file_with_embeddings(
        self, 
        input_path: Union[str, Path],
        overwrite_embeddings: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single file through the pipeline and generate embeddings.
        """
        # First, process the file to get chunks
        chunks = self.process_file(input_path)
        
        if not chunks:
            return {"chunks": [], "embeddings": []}
        
        # Generate embeddings for the chunks
        source_name = Path(input_path).stem
        embeddings = self.generate_embeddings(
            chunks=chunks,
            source_name=source_name,
            overwrite=overwrite_embeddings
        )
        
        return {
            "chunks": chunks,
            "embeddings": embeddings
        }
    
    def verify_embeddings_storage(self):
        """Verify that only embeddings are stored in Qdrant."""
        print(f"\n🔍 Verifying embeddings-only storage...")
        
        try:
            # Get collection info
            collection_info = self.vector_store.client.get_collection(self.vector_store.collection_name)
            points_count = self.vector_store.client.count(collection_name=self.vector_store.collection_name)
            
            print(f"✅ Collection '{self.vector_store.collection_name}' exists")
            print(f"📊 Total embeddings stored: {points_count.count}")
            
            if points_count.count > 0:
                # Get a sample
                points, _ = self.vector_store.client.scroll(
                    collection_name=self.vector_store.collection_name,
                    limit=2,
                    with_payload=True,
                    with_vectors=True
                )
                
                print(f"\n📝 Sample embeddings (no text stored):")
                for i, point in enumerate(points):
                    print(f"  {i+1}. ID: {point.id}")
                    print(f"     Vector dimensions: {len(point.vector)}")
                    print(f"     Metadata keys: {list(point.payload.keys())}")
                    # Verify no text content is stored
                    if 'text' not in point.payload and 'content' not in point.payload:
                        print(f"     ✅ No text content stored")
                    else:
                        print(f"     ❌ Text content found: {point.payload.get('text', '')[:50]}...")
                
                return True
            else:
                print("❌ No embeddings found in collection!")
                return False
                
        except Exception as e:
            print(f"❌ Error verifying storage: {e}")
            return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Document Processing Pipeline - Parse, redact, and chunk documents"
    )
    
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to a file or directory to process"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="chunking_output",
        help="Directory to save processed files (default: chunking_output/)"
    )
    
    parser.add_argument(
        "--no-redact",
        action="store_false",
        dest="redact_sensitive",
        help="Disable redaction of sensitive information"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum size of each chunk in tokens (default: 1000)"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Number of tokens to overlap between chunks (default: 200)"
    )
    
    parser.add_argument(
        "--pattern",
        type=str,
        default="*",
        help="File pattern to match when processing directories (default: *)"
    )
    
    parser.add_argument(
        "--embed-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Name of the embedding model to use (default: all-MiniLM-L6-v2)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default='cuda' if os.environ.get("CUDA_VISIBLE_DEVICES") else 'cpu',
        help='Device to run the model on (cuda or cpu)'
    )
    
    parser.add_argument(
        "--qdrant-collection",
        type=str,
        default='document_embeddings',
        help='Name of the Qdrant collection to store vectors'
    )
    
    parser.add_argument(
        "--recreate-collection",
        action='store_true',
        help='Recreate the Qdrant collection if it exists'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_arguments()
    
    # Initialize the processor
    processor = DocumentProcessor(
        output_dir=args.output_dir,
        redact_sensitive=args.redact_sensitive,
        max_chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        file_pattern=args.pattern,
        embed_model=args.embed_model,
        device=args.device,
        qdrant_collection=args.qdrant_collection,
        recreate_collection=args.recreate_collection
    )
    
    # Process the input path
    input_path = Path(args.input_path)
    
    if input_path.is_file():
        processor.process_file(input_path)
        # Verify embeddings-only storage
        processor.verify_embeddings_storage()
    elif input_path.is_dir():
        processor.process_directory(input_path)
        # Verify embeddings-only storage
        processor.verify_embeddings_storage()
    else:
        print(f"Error: {input_path} is not a valid file or directory", file=sys.stderr)
        sys.exit(1)
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()