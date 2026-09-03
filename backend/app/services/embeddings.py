"""
Embedding Service
Handles text embedding generation using Sentence Transformers
"""
import logging
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.embedding_model
        self.model: Optional[SentenceTransformer] = None
        self.dimension = settings.embedding_dimension

    def load_model(self):
        """Load the embedding model (lazy loading)."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Dimension: {self.dimension}")

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        self.load_model()
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Embed multiple documents efficiently."""
        self.load_model()
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings.tolist()

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        self.load_model()
        return self.dimension

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        a_arr = np.array(a)
        b_arr = np.array(b)
        return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


# Global instance
embedding_service = EmbeddingService()
