# test_qdrant_standalone.py
# 100% FIXED for qdrant-client==1.16.1 + local Qdrant binary

from qdrant_client import QdrantClient, models

client = QdrantClient("http://localhost:6333")

collection_name = "standalone_test"

print("Qdrant standalone test starting...\n")

# 1. Delete if exists
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)
    print(f"Deleted old collection: {collection_name}")

# 2. Create collection
client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
)
print(f"Created collection: {collection_name}")

# 3. Insert 3 dummy points
client.upsert(
    collection_name=collection_name,
    points=[
        models.PointStruct(id=1, vector=[0.1, 0.2, 0.3, 0.4], payload={"text": "cat sits on mat"}),
        models.PointStruct(id=2, vector=[0.9, 0.8, 0.7, 0.6], payload={"text": "dog runs in park"}),
        models.PointStruct(id=3, vector=[0.2, 0.1, 0.4, 0.3], payload={"text": "kitten plays with yarn"}),
    ]
)
print("Inserted 3 points")

# 4. SEARCH — FIXED: Use plain list for query (auto-converts to VectorQuery in v1.16.1)
results = client.query_points(
    collection_name=collection_name,
    query=[0.15, 0.15, 0.35, 0.35],  # ← Plain list, no QueryVector needed!
    limit=3,
    with_payload=True
)

print("\nSearch results:")
for hit in results.points:
    print(f"  Score: {hit.score:.4f} → {hit.payload['text']}")

# 5. Count points
count = client.count(collection_name=collection_name)
print(f"\nCollection has {count.count} points")

print("\nQDRANT IS 100% WORKING!")
print("You are now ready for EchoHybrid ingestion & retrieval")