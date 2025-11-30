import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class QdrantVectorStore:
    """A class to handle vector storage and retrieval using Qdrant."""
    
    def __init__(
        self, 
        collection_name: str = "document_embeddings",
        vector_size: int = 384,  # Default for all-MiniLM-L6-v2
        recreate_collection: bool = False
    ):
        """
        Initialize Qdrant client and collection.
        
        Args:
            collection_name: Name of the collection to store vectors
            vector_size: Dimensionality of the vectors
            recreate_collection: Whether to recreate the collection if it exists
        """
        # Get Qdrant configuration from environment variables
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            url=f"http://{self.qdrant_host}:{self.qdrant_port}"
        )
        
        # Create or get collection
        self._setup_collection(recreate=recreate_collection)
    
    def _setup_collection(self, recreate: bool = False):
        """Set up the Qdrant collection."""
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name in collection_names:
            if recreate:
                self.client.delete_collection(collection_name=self.collection_name)
            else:
                return
        
        # Create new collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE
            )
        )
    
    def add_embeddings(
        self, 
        embeddings: List[List[float]], 
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Add ONLY embeddings to the vector store.
        
        Args:
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries (without text content)
            
        Returns:
            List of document IDs
        """
        if metadatas is None:
            metadatas = [{}] * len(embeddings)
        
        # Generate proper UUIDs
        ids = [str(uuid.uuid4()) for _ in embeddings]
        
        # Prepare points for Qdrant - store embeddings and full metadata including text
        points = []
        for idx, (embedding, metadata) in enumerate(zip(embeddings, metadatas)):
            # Include all metadata, including text content
            point = PointStruct(
                id=ids[idx],
                vector=embedding,
                payload=metadata  # Store all metadata including text
            )
            points.append(point)
        
        # Upload to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        return ids
    
    def similarity_search(
        self, 
        query_embedding: List[float], 
        k: int = 4,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the collection.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filter_conditions: Optional filter conditions
            
        Returns:
            List of search results with metadata and score
        """
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k,
            query_filter=filter_conditions
        )
        
        results = []
        for hit in search_result:
            result = {
                "embedding": hit.vector if hasattr(hit, 'vector') else None,
                "metadata": hit.payload,  # Only metadata, no text
                "score": hit.score,
                "id": hit.id
            }
            results.append(result)
        
        return results
    
    def delete_embeddings(self, ids: List[str]) -> bool:
        """
        Delete embeddings by their IDs.
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            True if successful
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=ids
        )
        return True
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        return self.client.get_collection(collection_name=self.collection_name)