"""
Main RAG Pipeline
Orchestrates the complete retrieval-augmented generation pipeline with:
- Docling-based document parsing
- Adaptive chunking (semantic, parent-child, structured, fixed)
- AMER multi-stage embedding retrieval
- Hybrid search, reranking, MMR diversity
"""
import logging
import time
import json
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from app.config import settings
from app.services.hybrid_search import HybridSearchEngine
from app.services.reranker import CrossEncoderReranker, MaximalMarginalRelevance, ScoredDocument
from app.services.query_enhancer import QueryEnhancer, EnhancedQuery
from app.services.llm_service import llm_service
from app.services.memory import conversation_memory, ContextualRetriever
from app.services.vector_store import vector_store
from app.services.docling_parser import DoclingParser
from app.services.adaptive_chunker import AdaptiveChunker, AMERetriever, Chunk

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
    1. Docling document parsing (layout-aware, structured extraction)
    2. Adaptive chunking (section-aware, content-type aware)
    3. Query Enhancement (expansion, HyDE, multi-query)
    4. AMER multi-stage retrieval (coarse + clinical + RRF fusion)
    5. Cross-Encoder Reranking
    6. MMR Diversity Selection
    7. Context Assembly + LLM Generation with citations
    """

    def __init__(self):
        self.hybrid_search: Optional[HybridSearchEngine] = None
        self.reranker = CrossEncoderReranker()
        self.mmr = MaximalMarginalRelevance()
        self.query_enhancer = QueryEnhancer(llm_service)
        self.contextual_retriever = ContextualRetriever(conversation_memory)

        # New: Docling parser, adaptive chunker, AMER retriever
        self.docling_parser = DoclingParser()
        self.adaptive_chunker = AdaptiveChunker(
            base_chunk_size=settings.parent_chunk_size,
            chunk_overlap=settings.parent_chunk_overlap,
            parent_size=settings.parent_chunk_size,
            child_size=settings.child_chunk_size,
        )
        self.amer_retriever = AMERetriever(
            fast_embedding_model=settings.embedding_model,
            clinical_embedding_model=None,
        )

        # Stats
        self.total_queries = 0
        self.avg_latency = 0.0
        self.last_ingest_stats = {}

    def initialize(self, collection):
        self.hybrid_search = HybridSearchEngine(collection)
        logger.info("RAG Pipeline initialized with Docling + Adaptive Chunking + AMER")

    # ─── Document Ingestion (new) ───────────────────────────────────────────

    def ingest_documents(self) -> Dict:
        """
        Ingest all documents from the data directory.

        Pipeline:
        1. Docling parsing → structured elements
        2. Adaptive chunking → section-aware chunks
        3. Embedding + indexing → ChromaDB
        """
        docs_dir = Path(settings.documents_dir)
        processed_dir = Path(settings.processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)

        if not docs_dir.exists():
            return {"status": "error", "message": f"Documents directory not found: {docs_dir}"}

        # Find documents — support root-level and subdirectories
        supported_extensions = {'.pdf', '.pptx', '.docx', '.doc', '.txt', '.png', '.jpg', '.jpeg'}
        files = []
        for root, dirs, filenames in os.walk(docs_dir):
            for f in filenames:
                fp = Path(root) / f
                if fp.suffix.lower() in supported_extensions:
                    files.append(fp)

        if not files:
            return {"status": "warning", "message": "No documents found", "documents_processed": 0, "chunks_created": 0}

        all_chunks = []
        docs_processed = 0
        docs_failed = 0

        for filepath in sorted(files):
            try:
                logger.info(f"Ingesting: {filepath.name}")

                # Step 1: Docling parsing
                parsed_doc = self.docling_parser.parse_file(filepath)
                if not parsed_doc:
                    logger.warning(f"Skipped (no content): {filepath.name}")
                    continue

                # Step 2: Adaptive chunking
                chunks = self.adaptive_chunker.chunk_document(parsed_doc, strategy="adaptive")

                if not chunks:
                    logger.warning(f"No chunks from: {filepath.name}")
                    continue

                all_chunks.extend(chunks)
                docs_processed += 1

                # Save processed document
                processed_file = processed_dir / f"{parsed_doc.file_hash}_{filepath.stem}.json"
                self._save_processed_doc(parsed_doc, chunks, processed_file)

                logger.info(f"  ✓ {filepath.name}: {len(chunks)} chunks ({parsed_doc.elements} elements)")

            except Exception as e:
                docs_failed += 1
                logger.error(f"  ✗ Error processing {filepath.name}: {e}")

        # Step 3: Embed and index all chunks
        indexed = self._index_chunks(all_chunks)

        stats = {
            "status": "completed",
            "documents_processed": docs_processed,
            "documents_failed": docs_failed,
            "chunks_created": len(all_chunks),
            "chunks_indexed": indexed,
            "message": f"Ingested {docs_processed} documents → {len(all_chunks)} chunks → {indexed} indexed",
        }
        self.last_ingest_stats = stats
        logger.info(f"Ingestion complete: {stats['message']}")
        return stats

    def _index_chunks(self, chunks: List[Chunk]) -> int:
        """Embed and index chunks into ChromaDB."""
        if not chunks:
            return 0

        try:
            from app.services.embeddings import embedding_service
            texts = [c.text for c in chunks]
            ids = [c.chunk_id for c in chunks]
            metadatas = []
            for c in chunks:
                meta = {
                    "text": c.text,
                    "chunk_type": c.chunk_type,
                    "content_type": c.content_type,
                    "section": c.section,
                    "subsection": c.subsection,
                    "page": c.page_number,
                    "token_count": c.token_count,
                    "source": c.metadata.get("source", ""),
                    "file_type": c.metadata.get("file_type", ""),
                }
                if c.parent_id:
                    meta["parent_id"] = c.parent_id
                metadatas.append(meta)

            embeddings = embedding_service.embed_documents(texts)
            vector_store.add_documents(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=texts)
            return len(chunks)
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            return 0

    def _save_processed_doc(self, parsed_doc, chunks: List[Chunk], filepath: Path):
        """Save processed document for debugging/inspection."""
        try:
            import json
            data = {
                "filename": parsed_doc.filename,
                "title": parsed_doc.title,
                "file_hash": parsed_doc.file_hash,
                "metadata": parsed_doc.metadata,
                "full_text_length": len(parsed_doc.full_text),
                "total_elements": len(parsed_doc.elements),
                "chunks": [
                    {
                        "chunk_id": c.chunk_id,
                        "text": c.text[:200],
                        "chunk_type": c.chunk_type,
                        "content_type": c.content_type,
                        "token_count": c.token_count,
                        "section": c.section,
                        "parent_id": c.parent_id,
                    }
                    for c in chunks
                ],
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Could not save processed doc: {e}")

    # ─── Query Pipeline ──────────────────────────────────────────────────────

    def query(
        self,
        user_query: str,
        session_id: str = "default",
        top_k: int = None,
        use_hybrid: bool = True,
        use_reranking: bool = True,
        use_mmr: bool = True,
        use_amer: bool = True,
    ) -> RAGResult:
        """
        Execute RAG query with full pipeline.

        Args:
            use_amer: Enable AMER multi-stage retrieval instead of standard hybrid
        """
        start_time = time.time()
        self.total_queries += 1

        top_k = top_k or settings.top_k_final

        # Step 1: Conversation context
        conv_context = conversation_memory.get_conversation_context(session_id, max_turns=6)
        contextual_query = self.contextual_retriever.get_contextual_query(session_id, user_query)

        # Step 2: Query enhancement
        enhanced = self.query_enhancer.enhance(contextual_query, conv_context)

        # Step 3: Retrieval (AMER or standard hybrid)
        if use_amer:
            all_candidates = self._amer_retrieval(enhanced.expanded)
        else:
            all_candidates = self._standard_hybrid_retrieval(enhanced.expanded)

        # Step 3b: HyDE augmentation
        if enhanced.hyde_hypothetical:
            try:
                hyde_results = self._hyde_retrieval(enhanced.hyde_hypothetical)
                for doc_id, hyde_doc in hyde_results.items():
                    if not any(c.id == doc_id for c in all_candidates):
                        all_candidates.append(ScoredDocument(
                            id=doc_id, text=hyde_doc["text"], score=hyde_doc["score"] * 0.7,
                            metadata=hyde_doc["metadata"]
                        ))
            except Exception as e:
                logger.warning(f"HyDE retrieval failed: {e}")

        # Step 4: Reranking
        if use_reranking and len(all_candidates) > top_k:
            reranked = self.reranker.rerank(user_query, all_candidates, top_k=settings.top_k_rerank)
        else:
            reranked = sorted(all_candidates, key=lambda x: x.score, reverse=True)[:settings.top_k_rerank]

        # Step 5: MMR diversity
        if use_mmr and len(reranked) > top_k:
            final_docs = self.mmr.select(user_query, reranked, top_k=top_k)
        else:
            final_docs = reranked[:top_k]

        # Step 6: Context + generation
        context_parts, sources = self._build_context_and_sources(final_docs)
        answer = self._generate_answer(context_parts, sources, user_query)

        latency = (time.time() - start_time) * 1000
        self.avg_latency = (self.avg_latency * (self.total_queries - 1) + latency) / self.total_queries

        conversation_memory.add_user_message(session_id, user_query)
        conversation_memory.add_assistant_message(session_id, answer, sources=sources)

        return RAGResult(
            answer=answer, sources=sources,
            confidence=final_docs[0].score if final_docs else 0.0,
            retrieval_scores=[d.score for d in final_docs],
            enhanced_query=enhanced, latency_ms=round(latency, 1),
        )

    def _amer_retrieval(self, query: str) -> List[ScoredDocument]:
        """AMER retrieval → converts to ScoredDocument list."""
        try:
            results = self.amer_retriever.retrieve(query, vector_store, self.hybrid_search, top_k=settings.top_k_retrieval)
            candidates = []
            for doc in results:
                candidates.append(ScoredDocument(
                    id=doc["id"], text=doc.get("text", ""),
                    score=doc.get("score", 0.0), metadata=doc.get("metadata", {})
                ))
            return candidates
        except Exception as e:
            logger.warning(f"AMER retrieval failed: {e}, falling back to hybrid")
            return self._standard_hybrid_retrieval(query)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Direct document search without LLM generation."""
        candidates = self._amer_retrieval(query)
        if len(candidates) > top_k:
            reranked = self.reranker.rerank(query, candidates, top_k=top_k)
        else:
            reranked = sorted(candidates, key=lambda x: x.score, reverse=True)[:top_k]
        
        results = []
        for doc in reranked:
            results.append({
                "source": doc.metadata.get("source", ""),
                "page": doc.metadata.get("page"),
                "text": doc.text,
                "score": doc.score
            })
        return results

    def _standard_hybrid_retrieval(self, query: str) -> List[ScoredDocument]:
        """Standard hybrid retrieval."""
        all_candidates = []
        if self.hybrid_search:
            results = self.hybrid_search.search(query, top_k=settings.top_k_retrieval)
            for doc_id, fused_score, bm25_score, vec_score, metadata in results:
                all_candidates.append(ScoredDocument(
                    id=doc_id, text=metadata.get("text", ""), score=fused_score, metadata=metadata
                ))
        return all_candidates

    def _hyde_retrieval(self, hyde_query: str) -> Dict:
        """HyDE retrieval with AMER embedder."""
        try:
            from app.services.embeddings import embedding_service
            hyde_emb = self.amer_retriever.embed_query(hyde_query, stage="fast")
            results = vector_store.query(hyde_emb, top_k=10)
            docs = {}
            for i, doc_id in enumerate(results["ids"][0]):
                docs[doc_id] = {
                    "text": results["documents"][0][i],
                    "score": 0.5 / (1.0 + results["distances"][0][i]),
                    "metadata": results["metadatas"][0][i],
                }
            return docs
        except Exception as e:
            logger.warning(f"HyDE retrieval failed: {e}")
            return {}

    def _build_context_and_sources(self, final_docs: List[ScoredDocument]) -> Tuple[List[str], List[Dict]]:
        """Build context parts and source citations."""
        context_parts = []
        sources = []
        for i, doc in enumerate(final_docs):
            text = doc.text[:1200]
            context_parts.append(f"[Source {i+1}] {text}")
            sources.append({
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page"),
                "section": doc.metadata.get("section"),
                "score": round(doc.score, 3),
                "text_preview": text[:200],
                "content_type": doc.metadata.get("content_type", ""),
            })
        return context_parts, sources

    def _generate_answer(self, context_parts: List[str], sources: List[Dict], user_query: str) -> str:
        """Generate answer using LLM with clinical context."""
        context = "\n\n".join(context_parts)

        system_prompt = (
            "You are an expert Emergency Department Triage Assistant with deep knowledge of "
            "ESI (Emergency Severity Index), clinical assessment protocols, and emergency medicine. "
            "Provide accurate, evidence-based answers based on the provided context. "
            "Always cite sources using [Source N] format. If the context doesn't contain enough "
            "information, say so clearly. Prioritize patient safety in all recommendations."
        )

        prompt = (
            f"Context from clinical reference documents:\n{context}\n\n"
            f"User Question: {user_query}\n\n"
            "Instructions:\n"
            "- Answer based ONLY on the provided context\n"
            "- Cite sources using [Source N] format\n"
            "- If unsure, state that clearly\n"
            "- For ESI classification questions, explain the reasoning\n"
            "- Keep responses concise but thorough\n\n"
            "Answer:"
        )

        answer = llm_service.generate(prompt, system_prompt=system_prompt, max_tokens=800, temperature=0.1)
        return answer


rag_pipeline = RAGPipeline()
