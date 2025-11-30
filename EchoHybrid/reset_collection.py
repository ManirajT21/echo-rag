from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance  # ✅ needed imports

def reset_embeddings_collection():
    client = QdrantClient(host="localhost", port=6333)
    
    try:
        client.delete_collection(collection_name="document_embeddings")
        print("✅ Deleted existing collection")
    except Exception:
        print("ℹ️  No existing collection to delete")
    
    client.create_collection(
        collection_name="document_embeddings",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print("✅ Created new collection for embeddings only")

if __name__ == "__main__":
    reset_embeddings_collection()
