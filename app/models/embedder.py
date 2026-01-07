"""
Embedding model wrapper using sentence-transformers
"""
import numpy as np
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Wrapper for sentence-transformers embedding model.
    Converts text to dense vector representations for semantic similarity.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
                       Options:
                       - "all-MiniLM-L6-v2": Fast, 384 dims, good quality (recommended)
                       - "all-mpnet-base-v2": Slower, 768 dims, best quality
                       - "paraphrase-MiniLM-L6-v2": Fast, good for paraphrase detection
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.dimension: int = 0
        self._loaded = False
        
    def load(self) -> None:
        """Load the model into memory"""
        if self._loaded:
            return
            
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
        # Get embedding dimension from a test encoding
        test_embedding = self.model.encode("test", convert_to_numpy=True)
        self.dimension = len(test_embedding)
        
        self._loaded = True
        logger.info(f"Model loaded. Dimension: {self.dimension}")
        
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._loaded
        
    def encode(self, text: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Encode text(s) to embeddings.
        
        Args:
            text: Single text string or list of texts
            normalize: Whether to L2-normalize the embeddings (for cosine similarity)
            
        Returns:
            Numpy array of shape (embedding_dim,) for single text
            or (num_texts, embedding_dim) for list of texts
        """
        if not self._loaded:
            self.load()
            
        embeddings = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )
        
        return embeddings
    
    def encode_chunks(self, chunks: dict) -> dict:
        """
        Encode multiple named chunks.
        
        Args:
            chunks: Dict of chunk_name -> text
            
        Returns:
            Dict of chunk_name -> embedding (numpy array)
        """
        if not self._loaded:
            self.load()
            
        result = {}
        for name, text in chunks.items():
            if text and text.strip():
                result[name] = self.encode(text.strip())
            else:
                # Empty chunk gets zero vector
                result[name] = np.zeros(self.dimension)
                
        return result
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0 to 1 if normalized, -1 to 1 otherwise)
        """
        # If embeddings are normalized, dot product = cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        
        # Clamp to valid range in case of numerical errors
        return float(np.clip(dot_product, -1.0, 1.0))
    
    def batch_similarity(
        self, 
        query_embedding: np.ndarray, 
        candidate_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarities between query and multiple candidates.
        
        Args:
            query_embedding: Single query embedding (dim,)
            candidate_embeddings: Matrix of candidate embeddings (n, dim)
            
        Returns:
            Array of similarity scores (n,)
        """
        # Matrix multiplication for efficient batch similarity
        similarities = np.dot(candidate_embeddings, query_embedding)
        return np.clip(similarities, -1.0, 1.0)


# Global model instance (lazy-loaded)
_embedding_model: Optional[EmbeddingModel] = None


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingModel:
    """Get or create the global embedding model instance"""
    global _embedding_model
    
    if _embedding_model is None or _embedding_model.model_name != model_name:
        _embedding_model = EmbeddingModel(model_name)
        
    return _embedding_model
