"""
Main RAG Pipeline
Orchestrates the complete retrieval-augmented generation pipeline
"""
import logging
import time
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from app.config import settings
from app.services.hybrid_search import HybridSearchEngine
from app.services.reranker import CrossEncoderReranker, MaximalMarginalRelevance, ScoredDocument
from app.services.query_enhancer import QueryEnhancer, EnhancedQuery
from app.services.llm_service import llm_service
from app.services.memory import conversation_memory, ContextualRetriever
from app.services.vector_store import vector_store

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Result from RAG pipeline."""
    answer: str
    sources: List[Dict]
    confidence: float
    retrieval_scores: List[float]
    enhanced_query: Optional[EnhancedQuery] = None
    latency_ms: float = 0


class RAGPipeline:
    """
    Complete RAG pipeline with all advanced techniques:
    1. Query Enhancement (expansion, HyDE, multi-query)
    2. Hybrid Search (BM25 + Vector with RRF)
    3. Cross-Encoder Reranking
    4. MMR Diversity Selection
    5. Context Assembly
    6. LLM Generation with citations
    """

    def __init__(self):
        self.hybrid_search: Optional[HybridSearchEngine] = None
        self.reranker = CrossEncoderReranker()
        self.mmr = MaximalMarginalRelevance()
        self.query_enhancer = QueryEnhancer(llm_service)
        self.contextual_retriever = ContextualRetriever(conversation_memory)

        # Stats
        self.total_queries = 0
        self.avg_latency = 0.0

    def initialize(self, collection):
        self.hybrid_search = HybridSearchEngine(collection)
        logger.info("RAG Pipeline initialized")

    def query(
        self,
        user_query: str,
        session_id: str = "default",
        top_k: int = None,
        use_hybrid: bool = True,
        use_reranking: bool = True,
        use_mmr: bool = True,
    ) -> RAGResult:
        start_time = time.time()
        self.total_queries += 1

        top_k = top_k or settings.top_k_final

        # Step 1: Get conversation context
        conv_context = conversation_memory.get_conversation_context(session_id, max_turns=6)
        contextual_query = self.contextual_retriever.get_contextual_query(session_id, user_query)

        # Step 2: Enhance query
        enhanced = self.query_enhancer.enhance(contextual_query, conv_context)

        # Step 3: Hybrid retrieval
        all_candidates = []
        if use_hybrid and self.hybrid_search:
            hybrid_results = self.hybrid_search.search(enhanced.expanded, top_k=settings.top_k_retrieval)
            for doc_id, fused_score, bm25_score, vec_score, metadata in hybrid_results:
                all_candidates.append(ScoredDocument(
                    id=doc_id, text=metadata.get("text", ""), score=fused_score, metadata=metadata
                ))
        else:
            query_emb = embedding_service.embed_query(enhanced.expanded)
            results = vector_store.query(query_emb, top_k=settings.top_k_retrieval)
            for i, doc_id in enumerate(results["ids"][0]):
                all_candidates.append(ScoredDocument(
                    id=doc_id, text=results["documents"][0][i],
                    score=1.0 / (1.0 + results["distances"][0][i]),
                    metadata=results["metadatas"][0][i],
                ))

        # Step 3b: HyDE retrieval (augment with hypothetical)
        if enhanced.hyde_hypothetical:
            try:
                hyde_emb = embedding_service.embed_query(enhanced.hyde_hypothetical)
                hyde_results = vector_store.query(hyde_emb, top_k=10)
                for i, doc_id in enumerate(hyde_results["ids"][0]):
                    if not any(c.id == doc_id for c in all_candidates):
                        all_candidates.append(ScoredDocument(
                            id=doc_id, text=hyde_results["documents"][0][i],
                            score=0.5 / (1.0 + hyde_results["distances"][0][i]),
                            metadata=hyde_results["metadatas"][0][i],
                        ))
            except Exception as e:
                logger.warning(f"HyDE retrieval failed: {e}")

        # Step 4: Reranking
        if use_reranking and len(all_candidates) > top_k:
            reranked = self.reranker.rerank(user_query, all_candidates, top_k=settings.top_k_rerank)
        else:
            reranked = sorted(all_candidates, key=lambda x: x.score, reverse=True)[:settings.top_k_rerank]

        # Step 5: MMR diversity selection
        if use_mmr and len(reranked) > top_k:
            final_docs = self.mmr.select(user_query, reranked, top_k=top_k)
        else:
            final_docs = reranked[:top_k]

        # Step 6: Build context and generate answer
        context_parts = []
        sources = []
        for i, doc in enumerate(final_docs):
            text = doc.text[:800]
            context_parts.append(f"[Source {i+1}] {text}")
            sources.append({
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page"),
                "score": round(doc.score, 3),
                "text_preview": text[:200],
                "section": doc.metadata.get("section"),
            })

        context = "\n\n".join(context_parts)

        system_prompt = """You are an expert Emergency Department Triage Assistant with deep knowledge of ESI (Emergency Severity Index), clinical assessment protocols, and emergency medicine. Provide accurate, evidence-based answers based on the provided context. Always cite sources using [Source N] format. If the context doesn't contain enough information, say so clearly. Prioritize patient safety in all recommendations."""

        prompt = f"""Context from clinical reference documents:
{context}

User Question: {user_query}

Instructions:
- Answer based ONLY on the provided context
- Cite sources using [Source N] format
- If unsure, state that clearly
- For ESI classification questions, explain the reasoning
- Keep responses concise but thorough

Answer:"""

        answer = llm_service.generate(prompt, system_prompt=system_prompt, max_tokens=800, temperature=0.1)

        latency = (time.time() - start_time) * 1000
        self.avg_latency = (self.avg_latency * (self.total_queries - 1) + latency) / self.total_queries

        conversation_memory.add_user_message(session_id, user_query)
        conversation_memory.add_assistant_message(session_id, answer, sources=sources)

        return RAGResult(
            answer=answer,
            sources=sources,
            confidence=final_docs[0].score if final_docs else 0.0,
            retrieval_scores=[d.score for d in final_docs],
            enhanced_query=enhanced,
            latency_ms=round(latency, 1),
        )


rag_pipeline = RAGPipeline()
