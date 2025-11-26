import re
import tiktoken
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import json

# Import the cleaner
from .cleaner import clean_text, clean_chunks

@dataclass
class Chunk:
    """A class to represent a chunk of text with metadata."""
    text: str
    metadata: Dict[str, Any] = None
    chunk_type: str = "text"
    chunk_id: str = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the chunk to a dictionary."""
        return {
            "text": self.text,
            "metadata": self.metadata or {},
            "chunk_type": self.chunk_type,
            "chunk_id": self.chunk_id
        }

class HybridChunker:
    """
    A hybrid document chunker that combines semantic and token-based chunking.
    
    This chunker first tries to split documents into semantic units (like paragraphs, 
    sections, etc.) and falls back to token-based chunking when semantic units 
    are too large.
    """
    
    def __init__(
        self,
        max_chunk_size: int = 1000,
        chunk_overlap: int = 200,
        tokenizer_name: str = "cl100k_base"
    ):
        """
        Initialize the hybrid chunker.
        
        Args:
            max_chunk_size: Maximum size of each chunk in tokens
            chunk_overlap: Number of tokens to overlap between chunks
            tokenizer_name: Name of the tokenizer to use (default: cl100k_base for GPT-4)
        """
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        
        try:
            self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        except Exception as e:
            raise ImportError(
                "Could not import tiktoken. Please install it with: pip install tiktoken"
            ) from e
    
    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        return len(self.tokenizer.encode(text))
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split by double newlines, preserving the delimiters
        paragraphs = re.split(r'(\n\s*\n)', text)
        
        # Recombine the split parts (since split includes the delimiters)
        result = []
        for i in range(0, len(paragraphs), 2):
            para = paragraphs[i]
            if i + 1 < len(paragraphs):
                para += paragraphs[i+1]  # Add back the newlines
            if para.strip():
                result.append(para.strip())
        
        return result
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # This is a simple regex - might need adjustment based on your needs
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _chunk_by_tokens(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Split text into chunks based on token count."""
        # Tokenize the text
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        # Create chunks with overlap
        for i in range(0, len(tokens), self.max_chunk_size - self.chunk_overlap):
            # Get the chunk tokens
            chunk_tokens = tokens[i:i + self.max_chunk_size]
            
            # Convert back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Create chunk with metadata
            chunk = Chunk(
                text=chunk_text,
                metadata=metadata.copy(),
                chunk_type="token_based",
                chunk_id=f"chunk_{len(chunks) + 1}"
            )
            
            chunks.append(chunk)
            
            # Stop if we've processed all tokens
            if i + self.max_chunk_size >= len(tokens):
                break
                
        return chunks
    
    def _chunk_by_semantic_units(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Split text into chunks based on semantic units (paragraphs, sections, etc.).
        Falls back to token-based chunking for large semantic units.
        """
        # First try to split by paragraphs
        paragraphs = self._split_into_paragraphs(text)
        chunks = []
        
        for para in paragraphs:
            # If paragraph is too large, try to split by sentences
            if self._count_tokens(para) > self.max_chunk_size * 1.5:
                sentences = self._split_into_sentences(para)
                current_chunk = []
                current_size = 0
                
                for sent in sentences:
                    sent_size = self._count_tokens(sent)
                    
                    # If adding this sentence would exceed the max size, finalize the current chunk
                    if current_size + sent_size > self.max_chunk_size and current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        chunks.append(Chunk(
                            text=chunk_text,
                            metadata=metadata.copy(),
                            chunk_type="semantic_paragraph",
                            chunk_id=f"chunk_{len(chunks) + 1}"
                        ))
                        current_chunk = []
                        current_size = 0
                    
                    current_chunk.append(sent)
                    current_size += sent_size
                
                # Add the last chunk if there's any remaining text
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(Chunk(
                        text=chunk_text,
                        metadata=metadata.copy(),
                        chunk_type="semantic_paragraph",
                        chunk_id=f"chunk_{len(chunks) + 1}"
                    ))
            else:
                # If paragraph is a good size, add it as a chunk
                chunks.append(Chunk(
                    text=para,
                    metadata=metadata.copy(),
                    chunk_type="paragraph",
                    chunk_id=f"chunk_{len(chunks) + 1}"
                ))
        
        return chunks
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Chunk text into smaller pieces using a hybrid approach.
        
        Args:
            text: The text to chunk
            metadata: Optional metadata to include with each chunk
            
        Returns:
            List of chunk dictionaries
        """
        if not text or not isinstance(text, str):
            return []
            
        metadata = metadata or {}
        
        # Clean the text before chunking
        cleaned_text = clean_text(text)
        
        # First try semantic chunking
        chunks = self._chunk_by_semantic_units(cleaned_text, metadata)
        
        # If we only got one chunk and it's too large, fall back to token-based chunking
        if len(chunks) == 1 and self._count_tokens(chunks[0].text) > self.max_chunk_size * 1.5:
            chunks = self._chunk_by_tokens(cleaned_text, metadata)
        
        # Convert chunks to dictionaries and clean them
        result = [chunk.to_dict() for chunk in chunks]
        return clean_chunks(result)
    
    def chunk_document(self, document: Union[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk a document (either raw text or parsed document from parser).
        
        Args:
            document: Either a string (raw text) or a dictionary (from parser output)
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if isinstance(document, str):
            # If input is raw text, clean it first
            cleaned_text = clean_text(document)
            return self.chunk_text(cleaned_text, {"source": "raw_text", "chunking_strategy": "token_based"})
            
        elif isinstance(document, dict) and 'content' in document:
            # If input is from our parser
            metadata = document.get('metadata', {}).copy()
            
            # Clean the metadata values
            for key, value in metadata.items():
                if isinstance(value, str):
                    metadata[key] = clean_text(value)
            
            metadata.update({
                'file_type': document.get('file_type', 'unknown'),
                'chunking_strategy': 'hybrid'
            })
            
            # Clean and chunk the content
            cleaned_content = clean_text(document['content'])
            chunks = self.chunk_text(cleaned_content, metadata)
            return chunks
            
        else:
            raise ValueError("Unsupported document format. Must be either string or parsed document dict.")
    
    def chunk_file(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Chunk a file by first parsing it (if needed) and then chunking the content.
        
        Args:
            file_path: Path to the file to chunk
            
        Returns:
            List of chunk dictionaries
        """
        from .parser import DocumentParser
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # If it's a JSON file that might be from our parser
        if file_path.suffix.lower() == '.json':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    document = json.load(f)
                
                # Clean the document if it's a dictionary
                if isinstance(document, dict):
                    if 'content' in document and isinstance(document['content'], str):
                        document['content'] = clean_text(document['content'])
                    if 'metadata' in document and isinstance(document['metadata'], dict):
                        for key, value in document['metadata'].items():
                            if isinstance(value, str):
                                document['metadata'][key] = clean_text(value)
                
                return self.chunk_document(document)
            except json.JSONDecodeError:
                # If it's not a valid JSON, treat as raw text
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = clean_text(f.read())
                    return self.chunk_text(content, {"source": str(file_path)})
        else:
            # Use our parser for other file types
            parser = DocumentParser()
            document = parser.parse_document(file_path)
            
            # Clean the parsed document
            if isinstance(document, dict):
                if 'content' in document and isinstance(document['content'], str):
                    document['content'] = clean_text(document['content'])
                if 'metadata' in document and isinstance(document['metadata'], dict):
                    for key, value in document['metadata'].items():
                        if isinstance(value, str):
                            document['metadata'][key] = clean_text(value)
            
            return self.chunk_document(document)


def chunk_text(
    text: str,
    max_chunk_size: int = 1000,
    chunk_overlap: int = 200,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Convenience function to chunk text with default settings.
    
    Args:
        text: The text to chunk
        max_chunk_size: Maximum size of each chunk in tokens
        chunk_overlap: Number of tokens to overlap between chunks
        **kwargs: Additional metadata to include with each chunk
        
    Returns:
        List of chunk dictionaries
    """
    # Clean the input text and metadata
    cleaned_text = clean_text(text)
    cleaned_kwargs = {k: clean_text(v) if isinstance(v, str) else v 
                     for k, v in kwargs.items()}
    
    chunker = HybridChunker(
        max_chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap
    )
    return chunker.chunk_text(cleaned_text, cleaned_kwargs)


def chunk_file(
    file_path: Union[str, Path],
    max_chunk_size: int = 1000,
    chunk_overlap: int = 200,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Convenience function to chunk a file with default settings.
    
    Args:
        file_path: Path to the file to chunk
        max_chunk_size: Maximum size of each chunk in tokens
        chunk_overlap: Number of tokens to overlap between chunks
        **kwargs: Additional metadata to include with each chunk
        
    Returns:
        List of chunk dictionaries
    """
    # Clean any string metadata
    cleaned_kwargs = {k: clean_text(v) if isinstance(v, str) else v 
                     for k, v in kwargs.items()}
    
    # Create chunker with cleaned parameters
    chunker = HybridChunker(
        max_chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Process the file and clean the results
    chunks = chunker.chunk_file(file_path)
    
    # Add any additional metadata to each chunk
    if cleaned_kwargs:
        for chunk in chunks:
            if 'metadata' not in chunk:
                chunk['metadata'] = {}
            chunk['metadata'].update(cleaned_kwargs)
    
    return chunks
