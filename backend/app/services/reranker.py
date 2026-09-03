"""
Cross-Encoder Reranker
Uses a cross-encoder model to rerank retrieved documents for better relevance
"""
import logging
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ScoredDocument:
    """Document with relevance score."""
    id: str
    text: str
    score: float
    metadata: dict


class CrossEncoderReranker:
    """
    Cross-encoder reranking using sentence-transformers cross-encoder models.
    Cross-encoders jointly encode query+document pairs for better accuracy.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.reranker_model
        self.model = None

    def load_model(self):
        """Lazy load the reranker model."""
        if self.model is None:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder model loaded")

    def rerank(
        self,
        query: str,
        documents: List[ScoredDocument],
        top_k: int = None
    ) -> List[ScoredDocument]:
        """
        Rerank documents using cross-encoder.
        Each (query, document) pair is scored jointly.
        """
        if not documents:
            return []

        self.load_model()
        top_k = top_k or settings.top_k_rerank

        # Prepare pairs for cross-encoder
        pairs = [(query, doc.text[:512]) for doc in documents]

        # Get cross-encoder scores
        scores = self.model.predict(pairs)

        # Normalize scores to 0-1 range
        scores = self._normalize_scores(scores)

        # Update document scores
        for doc, score in zip(documents, scores):
            doc.score = float(score)

        # Sort by new scores and return top_k
        reranked = sorted(documents, key=lambda x: x.score, reverse=True)
        return reranked[:top_k]

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to 0-1 range using sigmoid."""
        return 1.0 / (1.0 + np.exp(-scores))

    def rerank_with_threshold(
        self,
        query: str,
        documents: List[ScoredDocument],
        top_k: int = None,
        threshold: float = 0.3
    ) -> List[ScoredDocument]:
        """
        Rerank with relevance threshold filtering.
        Only returns documents above the threshold score.
        """
        reranked = self.rerank(query, documents, top_k)
        return [doc for doc in reranked if doc.score >= threshold]


class MaximalMarginalRelevance:
    """
    MMR (Maximal Marginal Relevance) selection for diverse results.
    Balances relevance with diversity to avoid redundant information.
    """

    def __init__(self, lambda_param: float = None):
        self.lambda_param = lambda_param or settings.mmr_lambda

    def select(
        self,
        query: str,
        documents: List[ScoredDocument],
        top_k: int = None,
        already_selected: List[str] = None
    ) -> List[ScoredDocument]:
        """
        Select documents using MMR.
        lambda_param: 0 = maximum diversity, 1 = maximum relevance
        """
        top_k = top_k or settings.top_k_final
        already_selected = already_selected or []
        selected = []
        remaining = documents.copy()

        # Get query embedding for diversity calculation
        query_emb = np.array(embedding_service.embed_query(query))

        while len(selected) < top_k and remaining:
            mmr_scores = []

            for doc in remaining:
                # Relevance: similarity to query
                doc_emb = np.array(embedding_service.embed_query(doc.text[:512]))
                relevance = float(np.dot(query_emb, doc_emb))

                # Diversity: maximum similarity to already selected
                if selected:
                    doc_embs = [
                        np.array(embedding_service.embed_query(s.text[:512]))
                        for s in selected
                    ]
                    max_sim = max(
                        float(np.dot(doc_emb, de)) for de in doc_embs
                    )
                else:
                    max_sim = 0.0

                # MMR formula
                mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim
                mmr_scores.append((doc, mmr_score))

            # Pick the best MMR score
            best_doc, best_score = max(mmr_scores, key=lambda x: x[1])
            selected.append(best_doc)
            remaining.remove(best_doc)

        return selected
