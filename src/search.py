from typing import List, Dict, Any
from .vector_store import VectorStoreManager
from .embedding import EmbeddingManager

class Retriever:
    def __init__(self, vector_store: VectorStoreManager, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieves relevant documents based on the query."""
        
        # 1. Generate embedding for the query
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        # 2. Query the vector store
        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        # 3. Parse results
        retrieved_docs = []
        if results['documents'] and results['documents'][0]:
            # Chroma returns lists of lists
            docs = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            ids = results['ids'][0]

            for i, (doc_text, metadata, distance, doc_id) in enumerate(zip(docs, metadatas, distances, ids)):
                # Convert distance to a similarity score (approximate)
                similarity_score = 1 - distance 
                
                retrieved_docs.append({
                    "id": doc_id,
                    "content": doc_text,
                    "metadata": metadata,
                    "similarity_score": similarity_score,
                    "rank": i + 1
                })

        return retrieved_docs