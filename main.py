import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Import modular classes
from src.data_loader import DataLoader
from src.embedding import EmbeddingManager
from src.vector_store import VectorStoreManager
from src.search import Retriever
from src.pipeline import RAGPipeline

load_dotenv()

def main():
    # --- Configuration ---
    DATA_DIR = "./data"
    VECTOR_DB_DIR = "./data/vector_store"
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY not found in .env")
        return

    # 1. Setup Data & Search
    print("--- Initializing System ---")
    loader = DataLoader(DATA_DIR)
    embed_manager = EmbeddingManager()
    vector_store = VectorStoreManager(persist_directory=VECTOR_DB_DIR)

    # Optional: Load data if DB is empty
    raw_docs = loader.load_documents()
    chunks = loader.chunk_documents(raw_docs)
    if chunks:
        embeddings = embed_manager.generate_embeddings([doc.page_content for doc in chunks])
        vector_store.add_documents(chunks, embeddings)

    retriever = Retriever(vector_store, embed_manager)
   
    # 2. Setup LLM
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY, 
        model="llama-3.1-8b-instant", 
        temperature=0.1
    )

    # 3. Initialize Pipeline
    pipeline = RAGPipeline(retriever, llm)

    # --- Interactive Loop ---
    print("\n>>> Advanced RAG System Ready. (Type 'exit' to quit)")
    
    while True:
        query = input("\nUser Query: ")
        if query.lower() in ['exit', 'quit']:
            break

        # Run Pipeline
        results = pipeline.query(
            query, 
            top_k=5, 
            min_score=0.2, 
            stream=True, 
            summarize=True
        )

        print("\n" + "="*50)
        print(f"AI Response:\n{results['answer']}")
        
        if results['summary']:
            print(f"\nSummary:\n{results['summary']}")
            
        print("="*50 + "\n")

if __name__ == "__main__":
    main()