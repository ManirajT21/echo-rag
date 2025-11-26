# reset_collection.py
from qdrant_client import QdrantClient

def reset_embeddings_collection():
    client = QdrantClient(host="localhost", port=6333)
    
    try:
        # Delete existing collection
        client.delete_collection(collection_name="document_embeddings")
        print("✅ Deleted existing collection")
    except:
        print("ℹ️  No existing collection to delete")
    
    # Create new collection for embeddings only
    client.create_collection(
        collection_name="document_embeddings",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print("✅ Created new collection for embeddings only")

if __name__ == "__main__":
    reset_embeddings_collection()