import os
import chromadb
from chromadb.utils import embedding_functions
import pypdf

# 1. Setup paths
DATA_DIR = "data"
JURISDICTIONS = ["EU", "US", "UK"]

# 2. Initialize ChromaDB persistent client (saves data locally to disk)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# 3. Setup embedding function using a robust, fast sentence-transformer model
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# 4. Create or fetch our collection
collection = chroma_client.get_or_create_collection(
    name="ai_regulations", 
    embedding_function=embedding_func
)

def get_pdf_text_chunks(file_path, chunk_size=1500, overlap=200):
    chunks = []
    try:
        reader = pypdf.PdfReader(file_path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
        
        start = 0
        while start < len(full_text):
            end = start + chunk_size
            chunks.append(full_text[start:end])
            start += chunk_size - overlap
    except Exception:
        pass # Ignore unreadable files from our list
    return chunks

if __name__ == "__main__":
    print("⏳ Generating embeddings and populating ChromaDB (This may take a minute)...")
    
    doc_id_counter = 0
    
    for jurisdiction in JURISDICTIONS:
        folder_path = os.path.join(DATA_DIR, jurisdiction)
        if not os.path.exists(folder_path):
            continue
            
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".pdf"):
                file_path = os.path.join(folder_path, file_name)
                chunks = get_pdf_text_chunks(file_path)
                
                if not chunks:
                    continue
                
                # Prepare batches for ChromaDB Ingestion
                documents = []
                metadatas = []
                ids = []
                
                for idx, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadatas.append({
                        "source": file_name,
                        "jurisdiction": jurisdiction
                    })
                    ids.append(f"doc_{jurisdiction}_{doc_id_counter}_{idx}")
                
                doc_id_counter += 1
                
                # Upsert to ChromaDB vector collection
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"✅ Embedded and stored {len(chunks)} chunks from {file_name}")

    print(f"\n🏁 ChromaDB Initialization Complete! Current collection count: {collection.count()} chunks stored.")