"""
Hybrid Search Engine
Combines BM25 (sparse) and Vector (dense) search with Reciprocal Rank Fusion
"""
import logging
import re
import math
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np
from rank_bm25 import BM25Okapi
from app.services.embeddings import embedding_service
from app.config import settings

logger = logging.getLogger(__name__)


class BM25SearchEngine:
    """BM25 sparse retrieval engine."""

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        self.metadata: Dict[str, dict] = {}

    def index(self, documents: List[str], doc_ids: List[str], metadata_list: List[dict]):
        """Build BM25 index from documents."""
        self.documents = documents
        self.doc_ids = doc_ids
        self.metadata = {doc_id: meta for doc_id, meta in zip(doc_ids, metadata_list)}

        # Tokenize documents
        tokenized = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index built with {len(documents)} documents")

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float, dict]]:
        """Search BM25 index. Returns (doc_id, score, metadata)."""
        if not self.bm25:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top_k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = scores[idx]
            if score > 0:
                doc_id = self.doc_ids[idx]
                results.append((doc_id, float(score), self.metadata.get(doc_id, {})))

        return results

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        # Lowercase, split on whitespace, remove punctuation
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        # Remove stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                     'from', 'as', 'into', 'about', 'through', 'during', 'before',
                     'after', 'above', 'below', 'between', 'out', 'off', 'over',
                     'under', 'again', 'further', 'then', 'once', 'and', 'but',
                     'or', 'nor', 'not', 'so', 'very', 'just', 'because', 'but',
                     'and', 'if', 'or', 'that', 'this', 'these', 'those', 'it',
                     'its', 'they', 'them', 'their', 'he', 'she', 'we', 'our'}
        return [t for t in tokens if t not in stopwords and len(t) > 1]


class VectorSearchEngine:
    """Dense vector search engine using embeddings."""

    def __init__(self, collection):
        self.collection = collection

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float, dict]]:
        """Search vector collection. Returns (doc_id, score, metadata)."""
        query_embedding = embedding_service.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        if not results['ids'] or not results['ids'][0]:
            return []

        output = []
        for i, doc_id in enumerate(results['ids'][0]):
            # Convert distance to similarity score
            distance = results['distances'][0][i]
            score = 1.0 / (1.0 + distance)  # Normalize to 0-1 range
            metadata = results['metadatas'][0][i] if results['metadatas'] else {}
            output.append((doc_id, score, metadata))

        return output


class HybridSearchEngine:
    """
    Hybrid search combining BM25 and Vector search using Reciprocal Rank Fusion (RRF).
    RRF formula: score(d) = sum over all rankings of 1/(k + rank(d))
    """

    def __init__(self, collection, k: int = 60):
        self.bm25 = BM25SearchEngine()
        self.vector = VectorSearchEngine(collection)
        self.rrf_k = k  # RRF constant (higher = more emphasis on consistency)

    def build_index(self, documents: List[str], doc_ids: List[str], metadata_list: List[dict]):
        """Build both BM25 and vector indices."""
        self.bm25.index(documents, doc_ids, metadata_list)

    def search(self, query: str, top_k: int = None) -> List[Tuple[str, float, dict, dict]]:
        """
        Hybrid search with RRF fusion.
        Returns (doc_id, fused_score, bm25_score, vector_score, metadata).
        """
        top_k = top_k or settings.top_k_retrieval

        # Get results from both engines
        bm25_results = self.bm25.search(query, top_k=top_k)
        vector_results = self.vector.search(query, top_k=top_k)

        # Reciprocal Rank Fusion
        rrf_scores = defaultdict(float)
        doc_metadata = {}
        doc_bm25 = {}
        doc_vector = {}

        for rank, (doc_id, score, metadata) in enumerate(bm25_results):
            rrf_scores[doc_id] += 1.0 / (self.rrf_k + rank + 1)
            doc_metadata[doc_id] = metadata
            doc_bm25[doc_id] = score

        for rank, (doc_id, score, metadata) in enumerate(vector_results):
            rrf_scores[doc_id] += 1.0 / (self.rrf_k + rank + 1)
            doc_metadata[doc_id] = doc_metadata.get(doc_id, metadata)
            doc_vector[doc_id] = score

        # Sort by RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, fused_score in sorted_docs[:top_k]:
            results.append((
                doc_id,
                fused_score,
                doc_bm25.get(doc_id, 0.0),
                doc_vector.get(doc_id, 0.0),
                doc_metadata.get(doc_id, {}),
            ))

        return results
