import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple
import hashlib
import uuid
from dataclasses import dataclass, asdict, field
import tiktoken
from sentence_transformers import SentenceTransformer
from .vector_store import QdrantVectorStore

@dataclass
class EmbeddingResult:
    """A class to store embedding results with metadata."""
    text: str
    embedding: List[float]
    metadata: Dict[str, Any] = None
    chunk_id: str = None
    model_name: str = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'text': self.text,
            'embedding': self.embedding,
            'metadata': self.metadata or {},
            'chunk_id': self.chunk_id,
            'model_name': self.model_name
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingResult':
        """Create an EmbeddingResult from a dictionary."""
        return cls(
            text=data['text'],
            embedding=data['embedding'],
            metadata=data.get('metadata', {}),
            chunk_id=data.get('chunk_id'),
            model_name=data.get('model_name')
        )

class DocumentEmbedder:
    """
    A class to handle document embeddings using various models.
    """
    
    def __init__(
        self, 
        model_name: str = 'all-MiniLM-L6-v2',
        vector_store: Optional[QdrantVectorStore] = None,
        device: str = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
    ):
        """
        Initialize the document embedder.
        
        Args:
            model_name: Name of the embedding model to use
            vector_store: Optional QdrantVectorStore instance for storing embeddings
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device
        self.vector_store = vector_store
        
        # For backward compatibility
        self.output_dir = Path("generated_embeddings")
        
        self.model = self._load_model()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def _load_model(self):
        """Load the embedding model."""
        try:
            print(f"Loading model: {self.model_name}")
            return SentenceTransformer(self.model_name, device=self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {str(e)}")
    
    def _generate_chunk_id(self, text: str, metadata: Dict[str, Any]) -> str:
        """Generate a unique ID for a chunk based on its content and metadata."""
        unique_str = f"{text}:{json.dumps(metadata, sort_keys=True)}"
        return hashlib.md5(unique_str.encode()).hexdigest()
    
    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize embedding to unit length."""
        norm = np.linalg.norm(embedding)
        return (embedding / norm).tolist()
    
    def embed_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> EmbeddingResult:
        """
        Generate an embedding for a single text chunk.
        
        Args:
            text: The text to embed
            metadata: Optional metadata to include with the embedding
            
        Returns:
            EmbeddingResult containing the text, embedding, and metadata
        """
        if not text.strip():
            raise ValueError("Cannot embed empty text")
            
        metadata = metadata or {}
        chunk_id = self._generate_chunk_id(text, metadata)
        
        # Generate embedding
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            embedding = self._normalize_embedding(embedding)
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {str(e)}")
        
        return EmbeddingResult(
            text=text,
            embedding=embedding,
            metadata=metadata,
            chunk_id=chunk_id,
            model_name=self.model_name
        )
    
    def embed_batch(self, texts: List[str], metadata_list: Optional[List[Dict[str, Any]]] = None) -> List[EmbeddingResult]:
        """
        Generate embeddings for a batch of text chunks.
        
        Args:
            texts: List of text chunks to embed
            metadata_list: Optional list of metadata dictionaries
            
        Returns:
            List of EmbeddingResult objects
        """
        if not texts:
            return []
            
        if metadata_list is None:
            metadata_list = [{}] * len(texts)
        
        # Generate embeddings in a single batch for efficiency
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            results = []
            
            for i, (text, metadata, embedding) in enumerate(zip(texts, metadata_list, embeddings)):
                if not text.strip():
                    continue
                    
                chunk_id = self._generate_chunk_id(text, metadata)
                normalized_embedding = self._normalize_embedding(embedding)
                
                result = EmbeddingResult(
                    text=text,
                    embedding=normalized_embedding,
                    metadata=metadata,
                    chunk_id=chunk_id,
                    model_name=self.model_name
                )
                results.append(result)
                
            return results
            
        except Exception as e:
            raise RuntimeError(f"Batch embedding failed: {str(e)}")
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[EmbeddingResult]:
        """
        Generate embeddings for a list of chunk dictionaries.
        
        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata' keys
            
        Returns:
            List of EmbeddingResult objects
        """
        if not chunks:
            return []
            
        texts = [chunk.get('text', '') for chunk in chunks]
        metadata_list = [chunk.get('metadata', {}) for chunk in chunks]
        
        return self.embed_batch(texts, metadata_list)
    
    def embed_documents(
        self, 
        documents: List[Dict[str, Any]], 
        batch_size: int = 32,
        save_to_vector_store: bool = True,
        source_name: str = 'documents',
        collection_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Embed a list of document chunks and optionally store ONLY EMBEDDINGS in Qdrant.
        
        Args:
            documents: List of document chunks with 'text' and 'metadata' keys
            batch_size: Number of documents to process in each batch
            save_to_vector_store: Whether to save the embeddings to Qdrant
            source_name: Name of the source document (used for metadata)
            collection_name: Optional custom collection name for Qdrant
            
        Returns:
            List of dictionaries containing the text, embedding, and metadata
        """
        all_results = []
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            texts = [doc['text'] for doc in batch]
            metadatas = [doc.get('metadata', {}) for doc in batch]
            
            # Add source name to metadata
            for meta in metadatas:
                meta['source'] = meta.get('source', source_name)
            
            # Generate embeddings for the batch
            embedding_results = self.embed_batch(texts, metadatas)
            
            # Extract embeddings and create results as dictionaries
            batch_results = []
            embeddings_batch = []
            metadatas_for_qdrant = []
            
            for result in embedding_results:
                result_dict = {
                    'text': result.text,
                    'embedding': result.embedding,
                    'metadata': result.metadata,
                    'chunk_id': result.chunk_id,
                    'model_name': self.model_name
                }
                batch_results.append(result_dict)
                embeddings_batch.append(result.embedding)
                
                # Create metadata without text content for Qdrant
                metadata_without_text = {
                    k: v for k, v in result.metadata.items() 
                    if k not in ['text', 'content']
                }
                metadatas_for_qdrant.append(metadata_without_text)
            
            # Save to vector store if configured - ONLY EMBEDDINGS
            if save_to_vector_store and self.vector_store:
                self.vector_store.add_embeddings(
                    embeddings=embeddings_batch,
                    metadatas=metadatas_for_qdrant  # Metadata without text
                )
            
            all_results.extend(batch_results)
        
        return all_results
    
    def save_embeddings(
        self, 
        embeddings: List[EmbeddingResult], 
        source_name: str,
        overwrite: bool = False
    ) -> Path:
        """
        Save embeddings to a JSON file.
        
        Args:
            embeddings: List of EmbeddingResult objects
            source_name: Name of the source document
            overwrite: Whether to overwrite existing files
            
        Returns:
            Path to the saved embeddings file
        """
        if not embeddings:
            raise ValueError("No embeddings to save")
            
        output_file = Path("generated_embeddings") / f"{source_name}_{self.model_name}.json"
        
        if output_file.exists() and not overwrite:
            raise FileExistsError(f"Embeddings file {output_file} already exists. Set overwrite=True to replace it.")
        
        # Convert embeddings to serializable format
        serialized = [asdict(e) for e in embeddings]
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'model': self.model_name,
                'source': source_name,
                'embeddings': serialized,
                'count': len(embeddings)
            }, f, indent=2)
            
        return output_file
    
    def load_embeddings(self, source_name: str) -> List[EmbeddingResult]:
        """
        Load embeddings from a file.
        
        Args:
            source_name: Name of the source document
            
        Returns:
            List of EmbeddingResult objects
        """
        input_file = Path("generated_embeddings") / f"{source_name}_{self.model_name}.json"
        
        if not input_file.exists():
            raise FileNotFoundError(f"No embeddings found for {source_name}")
            
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        return [EmbeddingResult.from_dict(e) for e in data.get('embeddings', [])]


def embed_documents(
    documents: List[Dict[str, Any]],
    model_name: str = 'all-MiniLM-L6-v2',
    device: str = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu',
    save_to_vector_store: bool = True,
    source_name: str = 'documents',
    collection_name: Optional[str] = None,
    vector_store: Optional[QdrantVectorStore] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function to embed a list of document chunks.
    
    Args:
        documents: List of document chunks with 'text' and 'metadata' keys
        model_name: Name of the embedding model to use
        device: Device to run the model on ('cuda' or 'cpu')
        save_to_vector_store: Whether to save the embeddings to Qdrant
        source_name: Name of the source document
        collection_name: Optional custom collection name for Qdrant
        vector_store: Optional pre-configured QdrantVectorStore instance
        
    Returns:
        List of dictionaries containing the text, embedding, and metadata
    """
    if vector_store is None and save_to_vector_store:
        # Initialize default vector store if not provided
        vector_store = QdrantVectorStore(
            collection_name=collection_name or "document_embeddings",
            vector_size=384 if 'MiniLM' in model_name else 768
        )
    
    embedder = DocumentEmbedder(
        model_name=model_name,
        vector_store=vector_store,
        device=device
    )
    
    results = embedder.embed_documents(
        documents=documents,
        save_to_vector_store=save_to_vector_store,
        source_name=source_name,
        collection_name=collection_name
    )
    
    return results