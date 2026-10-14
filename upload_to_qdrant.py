import chromadb
from chromadb.utils import embedding_functions
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
import os
import time

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "ai_regulations"

# Connect to local ChromaDB
print("Connecting to local ChromaDB...")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
col = chroma_client.get_collection(name="ai_regulations", embedding_function=ef)
print(f"Found {col.count()} chunks in ChromaDB")

# Connect to Qdrant cloud with longer timeout
print("Connecting to Qdrant cloud...")
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60  # 60 second timeout instead of default
)

# Create collection in Qdrant (384 = MiniLM embedding size)
print("Creating Qdrant collection...")
try:
    qdrant.delete_collection(COLLECTION_NAME)
    print("Deleted old collection")
except:
    pass

qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)
print("Collection created")

# Upload in small batches of 25 to avoid timeout
BATCH_SIZE = 25
total = col.count()
uploaded = 0

print(f"Uploading {total} chunks to Qdrant in batches of {BATCH_SIZE}...")

for offset in range(0, total, BATCH_SIZE):
    batch = col.get(
        limit=BATCH_SIZE,
        offset=offset,
        include=["documents", "metadatas", "embeddings"]
    )

    if not batch["ids"]:
        break

    points = []
    for i, doc_id in enumerate(batch["ids"]):
        points.append(PointStruct(
            id=uploaded + i,
            vector=batch["embeddings"][i],
            payload={
                "text": batch["documents"][i],
                "source": batch["metadatas"][i].get("source", "unknown"),
                "jurisdiction": batch["metadatas"][i].get("jurisdiction", "unknown"),
                "source_file": batch["metadatas"][i].get("source", "unknown"),
            }
        ))

    # Retry up to 3 times if timeout
    for attempt in range(3):
        try:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            break
        except Exception as e:
            if attempt < 2:
                print(f"  Timeout on batch, retrying ({attempt+1}/3)...")
                time.sleep(3)
            else:
                print(f"  Failed after 3 attempts: {e}")
                raise

    uploaded += len(points)
    print(f"Uploaded {uploaded}/{total} chunks...")
    time.sleep(0.5)  # small pause between batches to avoid rate limiting

print(f"\nDone! {uploaded} chunks uploaded to Qdrant successfully.")

# Verify
count = qdrant.count(collection_name=COLLECTION_NAME)
print(f"Qdrant collection now has {count.count} points")