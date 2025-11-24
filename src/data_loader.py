import os
from pathlib import Path  
from typing import List
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DataLoader:
    def __init__(self, data_directory: str):
        self.data_directory = Path(data_directory)

    def load_documents(self) -> List[Document]:
        """Load documents from the specified directory"""
        all_documents = []

        if not self.data_directory.exists():
            print(f"Directory {self.data_directory} does not exist.")
            return []
        
        print(f"Loading documents from {self.data_directory}")

        # Load PDF files
        pdf_files = list(self.data_directory.glob("**/*.pdf"))
        print(f"Found {len(pdf_files)} PDF files.")
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source_file"] = pdf_file.name
                    doc.metadata["file_type"] = 'pdf'
                all_documents.extend(docs)
                print(f"Loaded PDF: {pdf_file.name} ({len(docs)} pages)")
            except Exception as e:
                print(f"Error loading PDF {pdf_file.name}: {e}")
        
        # Load TXT files
        txt_files = list(self.data_directory.glob("**/*.txt"))
        for txt_file in txt_files:
            try:
                loader = TextLoader(str(txt_file), encoding='utf-8')
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source_file"] = txt_file.name
                    doc.metadata["file_type"] = 'txt'
                all_documents.extend(docs)
                print(f"Loaded TXT: {txt_file.name} ({len(docs)} pages)")
            except Exception as e:
                print(f"Error loading TXT {txt_file.name}: {e}")
        
        print(f"Total documents loaded: {len(all_documents)}")
        return all_documents
    
    def chunk_documents(self, documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """Splits documents into smaller chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        split_docs = text_splitter.split_documents(documents) # call the split_documents method of the text splitter class
        print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
        return split_docs