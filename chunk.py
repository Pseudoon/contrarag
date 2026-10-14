"""
ContraRAG — Proper Semantic Chunking + Qdrant Upload
"""
import os, re, time
import pypdf
import spacy
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType

load_dotenv()

DATA_DIR = "data"
JURISDICTIONS = ["EU", "US", "UK"]
COLLECTION = "ai_regulations"
TARGET_CHUNK_WORDS = 150
OVERLAP_SENTENCES = 2
BATCH_SIZE = 25

print("Loading spacy...")
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000

print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Connecting to Qdrant...")
qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"), timeout=60)

try:
    qdrant.delete_collection(COLLECTION)
    print("Deleted old collection")
except:
    pass

qdrant.create_collection(collection_name=COLLECTION, vectors_config=VectorParams(size=384, distance=Distance.COSINE))
qdrant.create_payload_index(collection_name=COLLECTION, field_name="jurisdiction", field_schema=PayloadSchemaType.KEYWORD)
print("Fresh collection created")

def extract_text(path):
    text = ""
    try:
        if path.endswith(".pdf"):
            for page in pypdf.PdfReader(path).pages:
                t = page.extract_text()
                if t: text += t + " "
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
    except Exception as e:
        print(f"  Error: {e}")
    return text

def clean_text(text):
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'\n{2,}', '\n', text)
    return text.strip()

def semantic_chunk(text):
    doc = nlp(text)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    chunks, current_sents, current_words = [], [], 0
    for sent in sentences:
        current_sents.append(sent)
        current_words += len(sent.split())
        if current_words >= TARGET_CHUNK_WORDS:
            chunks.append(" ".join(current_sents))
            current_sents = current_sents[-OVERLAP_SENTENCES:]
            current_words = sum(len(s.split()) for s in current_sents)
    if current_sents:
        chunks.append(" ".join(current_sents))
    return chunks

all_chunks = []
doc_count = 0

for jurisdiction in JURISDICTIONS:
    folder = os.path.join(DATA_DIR, jurisdiction)
    if not os.path.exists(folder):
        continue
    files = [f for f in os.listdir(folder) if f.endswith((".pdf", ".txt"))]
    print(f"\n{jurisdiction}: {len(files)} files")
    for filename in files:
        print(f"  Processing {filename}...")
        raw = extract_text(os.path.join(folder, filename))
        if not raw:
            print("  Skipped (empty)")
            continue
        chunks = semantic_chunk(clean_text(raw))
        for i, chunk in enumerate(chunks):
            all_chunks.append({"text": chunk, "source": filename, "source_file": filename,
                               "jurisdiction": jurisdiction, "chunk_index": i, "doc_id": doc_count})
        print(f"  → {len(chunks)} chunks")
        doc_count += 1

print(f"\nTotal chunks: {len(all_chunks)}")
print("Embedding and uploading...")

uploaded = 0
for offset in range(0, len(all_chunks), BATCH_SIZE):
    batch = all_chunks[offset:offset+BATCH_SIZE]
    vectors = embed_model.encode([c["text"] for c in batch], show_progress_bar=False).tolist()
    points = [PointStruct(id=uploaded+i, vector=v, payload=c) for i,(c,v) in enumerate(zip(batch,vectors))]
    for attempt in range(3):
        try:
            qdrant.upsert(collection_name=COLLECTION, points=points)
            break
        except Exception as e:
            if attempt < 2:
                print(f"  Retry {attempt+1}/3...")
                time.sleep(3)
            else:
                raise
    uploaded += len(points)
    print(f"  Uploaded {uploaded}/{len(all_chunks)}")
    time.sleep(0.3)

print(f"\nDone! {uploaded} chunks uploaded")
print(f"Verified: {qdrant.count(collection_name=COLLECTION).count} points in Qdrant")