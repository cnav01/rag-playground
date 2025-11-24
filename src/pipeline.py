import time
from typing import List, Dict, Any

class RAGPipeline:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.history = []   # to store the history of queries

    def query(self, question: str, top_k: int = 5, min_score: float = 0.2, stream: bool = False, summarize: bool = False) -> Dict[str, Any]:
        # 1. Retrieve documents
        results = self.retriever.retrieve(question, top_k=top_k, score_threshold=min_score)
        
        if not results:
            return {
                'question': question,
                'answer': "No relevant documents found to answer the query.",
                'sources': [],
                'summary': None,
                'history': self.history
            }

        # 2. Format Context
        context = "\n\n".join([doc['content'] for doc in results])
        sources = [{
            'source': doc['metadata'].get('source_file', 'unknown'),
            'page': doc['metadata'].get('page', 'unknown'),
            'score': doc['similarity_score'],
            'preview': doc['content'][:100] + '....'
        } for doc in results] 

        # 3. Prepare Prompt
        prompt = f"""Use the following context to answer the question in a concise manner in paragraphs:
        
        Context:
        {context}
        
        Question: {question}"""

        # 4. Simulate Streaming (as per your code)
        if stream:
            print("Streaming response calculation...")
            # Note: This loops over the *prompt* just to simulate activity
            for i in range(0, len(prompt), 80):
                # print(prompt[i:i+80], end='', flush=True) # Optional: Don't print the raw prompt to user
                print(".", end='', flush=True) 
                time.sleep(0.05)
            print() 

        # 5. Generate Answer
        # Ensure we pass a string if that's what your specific LLM object expects, or a list
        response = self.llm.invoke(prompt)
        answer = response.content

        # 6. Add Citations
        citations = [f"[{i+1}] {src['source']} (Score: {src['score']:.2f})" for i, src in enumerate(sources)]
        answer_w_cite = answer + "\n\nCitations:\n" + "\n".join(citations) if citations else answer

        # 7. Summarize if required
        summary = None
        if summarize and answer:
            summary_prompt = f"Summarize the following answer in 2 or 3 sentences:\n\n{answer}"
            summary_response = self.llm.invoke(summary_prompt)
            summary = summary_response.content

        # 8. Store History
        entry = {
            'question': question,
            'answer': answer,
            'sources': sources,
            'summary': summary
        }
        self.history.append(entry)

        return {
            'question': question,
            'answer': answer_w_cite,    
            'sources': sources,
            'summary': summary,
            'history': self.history
        }