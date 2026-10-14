import os
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv  # 1. Import the environment loader

# 2. Load the .env file explicitly from your directory tree
load_dotenv()

# 3. Initialize persistent vector client matching your workspace folder
db_client = chromadb.PersistentClient(path="./chroma_db")
embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = db_client.get_collection(name="regulatory_corpus", embedding_function=embedding_model)

# 4. Initialize Groq client using your loaded environment keys
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def execute_contradiction_aware_rag(user_query):
    """
    Retrieves authentic statutory segments from ChromaDB using a stratified search 
    to guarantee cross-border representation, then passes them to Llama-3.
    """
    print(f"🔍 [Vector Search] Scanning database context matching: '{user_query}'...")
    
    retrieved_contexts = []
    print("\n📦 [Retrieved Chunks for Context Window]:")
    
    # 1. Stratified Search: Force retrieval from all 3 jurisdictions
    jurisdictions = ["EU", "US", "UK"]
    
    for region in jurisdictions:
        results = collection.query(
            query_texts=[user_query],
            n_results=2, # Get the top 2 best matches per region (6 total)
            where={"jurisdiction": region}
        )
        
        for idx, doc in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][idx]
            print(f"   -> Region: {meta['jurisdiction']} | Source: {meta['source_file']}")
            retrieved_contexts.append(f"Source Document: {meta['source_file']} ({meta['jurisdiction']})\nContent:\n{doc}\n---")

    context_window = "\n\n".join(retrieved_contexts)
    
    system_prompt = (
        "You are an expert AI compliance auditor evaluating global frameworks. "
        "Your task is to answer user queries using ONLY the provided regulatory context. "
        "CRITICAL: You will be provided with laws from the EU, US, and UK. You MUST explicitly "
        "compare them. Highlight direct or implicit contradictions (e.g., EU mandatory rules vs "
        "UK pro-innovation soft guidelines vs US state laws) and clearly note the regulatory friction."
    )
    
    user_prompt = f"""
    Context Data:
    {context_window}
    
    User Query: {user_query}
    
    Detailed Analysis:
    """
    
    print("\n🤖 [Groq API Inference] Sending federated context matrix to Llama-3...")
    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.1-8b-instant",  
            temperature=0.1          
        )
        
        output_text = response.choices[0].message.content
        print("\n======================= ANALYSIS OUTPUT =======================")
        print(output_text)
        print("===============================================================")
        return output_text
        
    except Exception as e:
        print(f"❌ Groq Inference Exception: {e}")
        return None

if __name__ == "__main__":
    # Test query designed to hit cross-border contradictions stored in your DB
    test_query = "What are the rules, mandatory disclosure thresholds, or definitions for general purpose AI and foundational models?"
    execute_contradiction_aware_rag(test_query)