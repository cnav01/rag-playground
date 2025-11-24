import os
import uuid
import chromadb
import numpy as np
from typing import Any, List
from langchain_core.documents import Document

class VectorStoreManager:
    def __init__(self, collection_name: str = "rag_collection", persist_directory: str = "../data/vector_store"):
        """
        Initialize the vector store

        Args:
            collection_name (str): Name of the vector store collection
            persist_directory (str): Directory to persist the vector store
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.collection = None 
        self.client = None 
        self._initialize_store()

    def _initialize_store(self):
        """Initialize ChromaDB client and collection"""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name, 
                metadata={"description": "RAG Document Embeddings"}
            ) 
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing deocuments in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Document], embeddings: np.ndarray): # embeddings is linked to previously created embeddings using the embedding manager function
        """
        Add documents and their embeddings to the vector store
        
        Args:
            documents (List[Document]): list of Document type objects
            embeddings (np.ndarray): Corresponding embeddings for the documents
        """    
        if len(documents) != embeddings.shape[0]:
            raise ValueError("Number of documents and embeddings must match.")
                
        # prepare data for chromadb
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # generate unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"  # it generates a unique id for each document using uuid module - 8 character hex string
            ids.append(doc_id)

            # prepare metadata
            meta = dict(doc.metadata)  # convert metadata to dictionary
            meta['doc_index'] = i # adding document index to metadata
            meta['content_length'] = len(doc.page_content)  # adding content length to metadata
            metadatas.append(meta)

            # document content
            documents_text.append(doc.page_content) # adding the actual text content of the document to the list

            # embeddings
            embeddings_list.append(embedding.tolist()) # convert numpy array to list and add to embeddings list

        # add to chromadb collection
        try:
            self.collection.add(   # calling add method of the Collection class
                ids=ids,
                metadatas=metadatas,
                documents=documents_text,
                embeddings=embeddings_list
            )
            print(f"Successfully added {len(documents)} chunks to the vector store.")

        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise