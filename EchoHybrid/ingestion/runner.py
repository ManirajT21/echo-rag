#!/usr/bin/env python3
"""
Document Processing Pipeline Runner

This script provides a command-line interface to the document processing pipeline,
which can parse various document formats, redact sensitive information, and chunk
the content into manageable pieces.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from ingestion.parser import DocumentParser, parse_and_save
from ingestion.chunker import HybridChunker
from ingestion.redactor import DocumentRedactor, RedactionRule


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
        file_pattern: str = "*"
    ):
        """
        Initialize the document processor.
        
        Args:
            output_dir: Directory to save processed files
            redact_sensitive: Whether to redact sensitive information
            max_chunk_size: Maximum size of each chunk in tokens
            chunk_overlap: Number of tokens to overlap between chunks
            file_pattern: File pattern to match when processing directories
        """
        self.output_dir = Path(output_dir)
        self.redact_sensitive = redact_sensitive
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.file_pattern = file_pattern
        
        # Initialize parser and chunker
        self.parser = DocumentParser(redact_sensitive=redact_sensitive)
        self.chunker = HybridChunker(
            max_chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Create output directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_file(self, input_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Process a single file through the pipeline.
        
        Args:
            input_path: Path to the input file
            
        Returns:
            List of chunked documents
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        print(f"\nProcessing file: {input_path}")
        
        # Step 1: Parse the document
        try:
            print("  - Parsing document...")
            parsed_doc = self.parser.parse_document(input_path)
            
            # Save the parsed output (for debugging)
            parsed_output_path = self.parser.get_output_path(input_path)
            self.parser.save_to_json(parsed_doc, input_path=input_path)
            print(f"  - Parsed output saved to: {parsed_output_path}")
            
            # Step 2: Chunk the document
            print("  - Chunking document...")
            chunks = self.chunker.chunk_document(parsed_doc)
            
            # Add source file information to each chunk
            for chunk in chunks:
                chunk_meta = chunk.get('metadata', {})
                chunk_meta.update({
                    'source_file': str(input_path),
                    'chunk_id': f"{input_path.stem}_{chunk_meta.get('chunk_id', '0')}"
                })
            
            # Save the chunks
            output_path = self._save_chunks(chunks, input_path)
            print(f"  - Chunks saved to: {output_path}")
            
            return chunks
            
        except Exception as e:
            print(f"Error processing {input_path}: {str(e)}", file=sys.stderr)
            return []
    
    def process_directory(self, input_dir: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Process all matching files in a directory.
        
        Args:
            input_dir: Directory containing files to process
            
        Returns:
            List of all chunked documents
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
        
        Args:
            chunks: List of chunk dictionaries
            input_path: Path to the input file (used to generate output filename)
            
        Returns:
            Path to the saved chunks file with timestamp
        """
        from datetime import datetime
        
        # Create a filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{input_path.stem}_chunks_{timestamp}.json"
        output_path = self.output_dir / output_filename
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the chunks
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        return output_path


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
        file_pattern=args.pattern
    )
    
    # Process the input path
    input_path = Path(args.input_path)
    
    if input_path.is_file():
        processor.process_file(input_path)
    elif input_path.is_dir():
        processor.process_directory(input_path)
    else:
        print(f"Error: {input_path} is not a valid file or directory", file=sys.stderr)
        sys.exit(1)
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
