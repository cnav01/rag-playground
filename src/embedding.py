from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class EmbeddingManager:
    """ Manages embedding models and their configurations."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model
        Args:
            model_name (str): HuggingFace model name for embeddings
        """
        print(f"Initializing Embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        Args:
            texts (List[str]): List of texts to embed
        Returns:
            np.ndarray: Array of embeddings
        """
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=False) # encoding using the sentence transformer model (encode is a method of SentenceTransformer class)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings