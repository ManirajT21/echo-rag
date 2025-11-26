# check_vectors.py
import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

def check_stored_vectors():
    # Initialize Qdrant client
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333"))
    )
    
    collection_name = "documents"
    
    try:
        # Check if collection exists
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if collection_name not in collection_names:
            print(f"❌ Collection '{collection_name}' does not exist!")
            return
        
        # Get collection info
        collection_info = client.get_collection(collection_name)
        print(f"✅ Collection: {collection_name}")
        
        # Get points count instead of vectors_count
        points_count = client.count(collection_name=collection_name)
        print(f"📊 Points count: {points_count.count}")
        
        # Get first few points to verify
        points, next_page_offset = client.scroll(
            collection_name=collection_name,
            limit=10,
            with_payload=True,
            with_vectors=False
        )
        
        print(f"\n📝 Sample of {len(points)} stored documents:")
        for i, point in enumerate(points):
            print(f"\n--- Document {i+1} ---")
            print(f"🆔 ID: {point.id}")
            text = point.payload.get('text', 'No text')
            print(f"📄 Text: {text[:150]}...")
            print(f"📋 Metadata:")
            for key, value in point.payload.items():
                if key != 'text':
                    print(f"   {key}: {value}")
                    
    except Exception as e:
        print(f"❌ Error checking vectors: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_stored_vectors()