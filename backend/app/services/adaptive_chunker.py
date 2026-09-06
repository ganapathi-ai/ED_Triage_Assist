"""
Adaptive Chunking Module
Intelligent chunking that adapts to document structure and content type.
"""
from __future__ import annotations
import logging
import re
from typing import List, Optional, Tuple, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum

if TYPE_CHECKING:
    from app.services.docling_parser import ParsedElement

logger = logging.getLogger(__name__)


class ChunkType(Enum):
    SEMANTIC = "semantic"
    PARENT_CHILD = "parent_child"
    FIXED = "fixed"
    RECURSIVE = "recursive"


class ContentType(Enum):
    NARRATIVE = "narrative"      # prose text → larger chunks
    STRUCTURED = "structured"    # tables, lists → smaller, precise chunks
    REFERENCE = "reference"      # protocols, guidelines → medium, section-aware
    DIALOGUE = "dialogue"        # Q&A, conversations → small chunks


@dataclass
class Chunk:
    """A text chunk with rich metadata for retrieval."""
    text: str
    chunk_id: str
    chunk_type: str
    content_type: str
    parent_id: Optional[str]
    metadata: dict = field(default_factory=dict)
    token_count: int = 0
    page_number: Optional[int] = None
    section: str = ""
    subsection: str = ""

    def __post_init__(self):
        if not self.metadata:
            self.metadata = {}


class AdaptiveChunker:
    """
    Adaptive chunking that adjusts strategy based on content type.

    Clinical document-aware:
    - Detects ESI/clinical sections → preserves boundaries
    - Tables → single-chunk per table row group
    - Protocols/guidelines → section-level parent chunks
    - Narratives → paragraph-level child chunks
    """

    def __init__(
        self,
        base_chunk_size: int = 512,
        chunk_overlap: int = 50,
        parent_size: int = 1024,
        child_size: int = 256,
        max_token_limit: int = 512,
    ):
        self.base_chunk_size = base_chunk_size
        self.chunk_overlap = chunk_overlap
        self.parent_size = parent_size
        self.child_size = child_size
        self.max_token_limit = max_token_limit

        # Clinical section headers for special handling
        self.clinical_sections = [
            r'assessment', r'protocol', r'guideline', r'procedure',
            r'criteria', r'diagnosis', r'treatment', r'management',
            r'intervention', r'monitoring', r'contraindicat',
            r'esi\s*(level|algorithm|triage)',
            r'abcde', r'primary survey', r'secondary survey',
            r'red flag', r'warning sign', r'discharge',
            r'medication', r'dosage', r'contraindication',
        ]

        # Try to use tiktoken for accurate token counting
        self._tokenizer = None
        try:
            import tiktoken
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            logger.warning("tiktoken not available, using word-based token estimation")

        # Try to use sentence-transformers tokenizer
        if not self._tokenizer:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            except Exception:
                logger.warning("No tokenizer available, using rough word count")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self._tokenizer:
            try:
                if hasattr(self._tokenizer, 'encode'):
                    return len(self._tokenizer.encode(text))
            except Exception:
                pass
        # Fallback: ~1.3 tokens per word
        return int(len(text.split()) * 1.3)

    def chunk_document(self, parsed_doc, strategy: str = "adaptive") -> List[Chunk]:
        """
        Chunk a parsed document using the specified strategy.

        Args:
            parsed_doc: ParsedDocument from docling_parser
            strategy: "adaptive", "parent_child", "semantic", "fixed"

        Returns:
            List of Chunk objects ready for embedding
        """
        if strategy == "parent_child":
            return self._parent_child_chunking(parsed_doc)
        elif strategy == "semantic":
            return self._semantic_chunking(parsed_doc)
        elif strategy == "fixed":
            return self._fixed_chunking(parsed_doc)
        else:
            return self._adaptive_chunking(parsed_doc)

    def _adaptive_chunking(self, parsed_doc) -> List[Chunk]:
        """
        Adaptive chunking: analyze content and choose optimal strategy per section.
        This is the default and recommended approach.
        """
        chunks = []
        elements = parsed_doc.elements

        # Group elements by section
        sections = self._group_by_section(elements)

        for section_name, section_elements in sections.items():
            content_type = self._detect_content_type(section_elements)

            if content_type == ContentType.STRUCTURED:
                # Tables/lists: keep each element as its own chunk
                chunks.extend(self._chunk_structured(section_elements, section_name, parsed_doc))
            elif content_type == ContentType.REFERENCE:
                # Protocols/guidelines: parent-child with section preservation
                chunks.extend(self._chunk_reference(section_elements, section_name, parsed_doc))
            elif content_type == ContentType.DIALOGUE:
                # Q&A: small, precise chunks
                chunks.extend(self._chunk_dialogue(section_elements, section_name, parsed_doc))
            else:
                # Narrative: standard semantic chunking
                chunks.extend(self._chunk_narrative(section_elements, section_name, parsed_doc))

        # Re-index chunks
        for i, chunk in enumerate(chunks):
            chunk.chunk_id = f"{parsed_doc.file_hash}_c{i}"
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)

        logger.info(f"Adaptive chunking: {len(chunks)} chunks from {len(elements)} elements ({len(sections)} sections)")
        return chunks

    def _detect_content_type(self, elements: List[ParsedElement]) -> ContentType:
        """Detect content type from elements."""
        types = [e.element_type for e in elements]
        text = " ".join(e.text for e in elements).lower()

        table_ratio = types.count('table') / max(len(types), 1)
        heading_ratio = types.count('heading') / max(len(types), 1)

        # Check for clinical reference patterns
        for pattern in self.clinical_sections:
            if re.search(pattern, text, re.IGNORECASE):
                return ContentType.REFERENCE

        if table_ratio > 0.3:
            return ContentType.STRUCTURED
        if heading_ratio > 0.3:
            return ContentType.REFERENCE
        if re.search(r'(q:|question:|a:|answer:)', text):
            return ContentType.DIALOGUE
        return ContentType.NARRATIVE

    def _group_by_section(self, elements: List[ParsedElement]) -> Dict[str, List[ParsedElement]]:
        """Group elements into sections based on headings."""
        sections = {}
        current_section = "Introduction"
        current_elements = []

        for elem in elements:
            if elem.element_type == 'heading':
                if current_elements:
                    sections.setdefault(current_section, []).extend(current_elements)
                    current_elements = []
                current_section = elem.text.strip()
            current_elements.append(elem)

        if current_elements:
            sections.setdefault(current_section, []).extend(current_elements)
        return sections

    def _chunk_structured(self, elements, section_name, doc) -> List[Chunk]:
        """Chunk structured content (tables) — each table is its own chunk."""
        chunks = []
        for elem in elements:
            if elem.element_type == 'table':
                tokens = self.count_tokens(elem.text)
                chunks.append(Chunk(
                    text=elem.text, chunk_id="", chunk_type="semantic",
                    content_type="structured", parent_id=None,
                    metadata={**elem.metadata, "section": section_name, "token_count": tokens},
                    token_count=tokens, page_number=elem.page_number, section=section_name
                ))
            elif elem.text.strip():
                tokens = self.count_tokens(elem.text)
                chunks.append(Chunk(
                    text=elem.text, chunk_id="", chunk_type="semantic",
                    content_type="structured", parent_id=None,
                    metadata={**elem.metadata, "section": section_name, "token_count": tokens},
                    token_count=tokens, page_number=elem.page_number, section=section_name
                ))
        return chunks

    def _chunk_reference(self, elements, section_name, doc) -> List[Chunk]:
        """Chunk reference content (protocols/guidelines) with parent-child."""
        chunks = []
        # Combine section text
        section_text = "\n\n".join(e.text for e in elements if e.text.strip())
        if not section_text.strip():
            return chunks

        # Create parent chunk (full section)
        parent_tokens = self.count_tokens(section_text)
        parent = Chunk(
            text=section_text, chunk_id="", chunk_type="parent_child",
            content_type="reference", parent_id=None,
            metadata={"section": section_name, "doc_id": doc.file_hash, "token_count": parent_tokens},
            token_count=parent_tokens, page_number=elements[0].page_number if elements else None,
            section=section_name
        )
        chunks.append(parent)

        # Create child chunks from the section
        child_size = self.child_size
        words = section_text.split()
        start = 0
        child_idx = 0

        while start < len(words):
            end = min(start + child_size, len(words))
            child_text = " ".join(words[start:end])
            child_tokens = self.count_tokens(child_text)

            child = Chunk(
                text=child_text, chunk_id="", chunk_type="parent_child",
                content_type="reference", parent_id=parent.chunk_id,
                metadata={"section": section_name, "doc_id": doc.file_hash,
                          "child_index": child_idx, "token_count": child_tokens},
                token_count=child_tokens, page_number=elements[0].page_number if elements else None,
                section=section_name
            )
            chunks.append(child)
            child_idx += 1
            start = end - self.chunk_overlap

        return chunks

    def _chunk_narrative(self, elements, section_name, doc) -> List[Chunk]:
        """Chunk narrative text using semantic boundaries."""
        return self._semantic_chunk_elements(elements, section_name, doc)

    def _chunk_dialogue(self, elements, section_name, doc) -> List[Chunk]:
        """Chunk dialogue/Q&A — each Q&A pair is a chunk."""
        chunks = []
        text = " ".join(e.text for e in elements if e.text.strip())

        # Split on Q&A patterns
        qa_pairs = re.split(r'(?=Q:|Question:|Q\.)', text)
        for pair in qa_pairs:
            pair = pair.strip()
            if len(pair.split()) < 10:
                continue
            tokens = self.count_tokens(pair)
            chunks.append(Chunk(
                text=pair, chunk_id="", chunk_type="semantic",
                content_type="dialogue", parent_id=None,
                metadata={"section": section_name, "doc_id": doc.file_hash, "token_count": tokens},
                token_count=tokens, section=section_name
            ))
        return chunks if chunks else self._semantic_chunk_elements(elements, section_name, doc)

    def _semantic_chunking(self, parsed_doc) -> List[Chunk]:
        """Pure semantic chunking based on document structure."""
        return self._semantic_chunk_elements(parsed_doc.elements, "Full Document", parsed_doc)

    def _semantic_chunk_elements(self, elements, section_name, doc) -> List[Chunk]:
        """Chunk elements at semantic boundaries."""
        chunks = []
        current_texts = []
        current_tokens = 0

        for elem in elements:
            if not elem.text.strip():
                continue

            elem_tokens = self.count_tokens(elem.text)

            # If single element exceeds limit, split it
            if elem_tokens > self.max_token_limit:
                if current_texts:
                    chunks.append(self._make_chunk(current_texts, section_name, doc))
                    current_texts, current_tokens = [], 0
                chunks.append(self._make_chunk([elem], section_name, doc))
                continue

            # If adding this element exceeds limit, flush current
            if current_tokens + elem_tokens > self.max_token_limit and current_texts:
                chunks.append(self._make_chunk(current_texts, section_name, doc))
                # Overlap: keep last element
                current_texts = [current_texts[-1], elem] if current_texts else [elem]
                current_tokens = sum(self.count_tokens(t.text) for t in current_texts)
            else:
                current_texts.append(elem)
                current_tokens += elem_tokens

        if current_texts:
            chunks.append(self._make_chunk(current_texts, section_name, doc))

        return chunks

    def _make_chunk(self, elements, section_name, doc) -> Chunk:
        """Create a chunk from a list of elements."""
        text = "\n\n".join(e.text for e in elements if e.text.strip())
        tokens = self.count_tokens(text)
        metadata = {
            "section": section_name,
            "doc_id": doc.file_hash,
            "token_count": tokens,
            "element_types": list(set(e.element_type for e in elements)),
        }
        if elements:
            metadata["page"] = elements[0].page_number
        return Chunk(
            text=text, chunk_id="", chunk_type="semantic",
            content_type="narrative", parent_id=None,
            metadata=metadata, token_count=tokens,
            page_number=elements[0].page_number if elements else None,
            section=section_name
        )

    def _fixed_chunking(self, parsed_doc) -> List[Chunk]:
        """Fixed-size chunking with overlap (baseline)."""
        chunks = []
        text = parsed_doc.full_text
        words = text.split()
        start = 0
        chunk_idx = 0

        while start < len(words):
            end = start + self.base_chunk_size
            chunk_text = " ".join(words[start:end])
            tokens = self.count_tokens(chunk_text)

            chunks.append(Chunk(
                text=chunk_text, chunk_id="", chunk_type="fixed",
                content_type="narrative", parent_id=None,
                metadata={"doc_id": parsed_doc.file_hash, "chunk_index": chunk_idx, "token_count": tokens},
                token_count=tokens
            ))
            chunk_idx += 1
            start = end - self.chunk_overlap

        return chunks

    def _parent_child_chunking(self, parsed_doc) -> List[Chunk]:
        """Traditional parent-child chunking (for comparison)."""
        chunks = []
        text = parsed_doc.full_text
        words = text.split()

        # Parent chunks
        start = 0
        parent_idx = 0
        while start < len(words):
            end = start + self.parent_size
            parent_text = " ".join(words[start:end])
            parent_tokens = self.count_tokens(parent_text)

            parent = Chunk(
                text=parent_text, chunk_id="", chunk_type="parent_child",
                content_type="reference", parent_id=None,
                metadata={"doc_id": parsed_doc.file_hash, "parent_index": parent_idx, "token_count": parent_tokens},
                token_count=parent_tokens
            )
            chunks.append(parent)

            # Child chunks
            child_start = 0
            child_idx = 0
            child_words = words[start:end]
            while child_start < len(child_words):
                child_end = child_start + self.child_size
                child_text = " ".join(child_words[child_start:child_end])
                child_tokens = self.count_tokens(child_text)

                chunks.append(Chunk(
                    text=child_text, chunk_id="", chunk_type="parent_child",
                    content_type="reference", parent_id=parent.chunk_id,
                    metadata={"doc_id": parsed_doc.file_hash, "parent_index": parent_idx,
                              "child_index": child_idx, "token_count": child_tokens},
                    token_count=child_tokens
                ))
                child_idx += 1
                child_start = child_end - 30

            parent_idx += 1
            start = end - 100

        return chunks


class AMERetriever:
    """
    AMER (Adaptive Multi-stage Embedding Retrieval) — multi-stage retrieval with
    stage-specific embedding models and adaptive fusion.

    Stages:
    1. Coarse retrieval: fast bi-encoder for broad recall
    2. Mid retrieval: domain-adapted embeddings for clinical precision
    3. Fine retrieval: cross-encoder reranking for top-k accuracy

    Features:
    - Stage-specific embedding models (fast vs accurate)
    - Reciprocal Rank Fusion (RRF) for combining stage results
    - Confidence-weighted fusion based on retrieval quality
    - Query-adaptive stage selection (simple queries → fewer stages)
    """

    def __init__(
        self,
        fast_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        clinical_embedding_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        fast_dim: int = 384,
        clinical_dim: int = 768,
    ):
        self.fast_model_name = fast_embedding_model
        self.clinical_model_name = clinical_embedding_model
        self.fast_dim = fast_dim
        self.clinical_dim = clinical_dim

        self._fast_embedder = None
        self._clinical_embedder = None
        self._cross_encoder = None

        self._init_embedders()

    def _init_embedders(self):
        """Initialize embedding models lazily."""
        try:
            from sentence_transformers import SentenceTransformer
            self._fast_embedder = SentenceTransformer(self.fast_model_name)
            logger.info(f"AMER fast embedder loaded: {self.fast_model_name}")
        except Exception as e:
            logger.warning(f"Could not load fast embedder: {e}")

        try:
            from sentence_transformers import SentenceTransformer
            self._clinical_embedder = SentenceTransformer(self.clinical_model_name)
            logger.info(f"AMER clinical embedder loaded: {self.clinical_model_name}")
        except Exception as e:
            logger.warning(f"Could not load clinical embedder: {e}")

    def embed_query(self, query: str, stage: str = "fast") -> List[float]:
        """Embed a query using the specified stage's model."""
        if stage == "clinical" and self._clinical_embedder:
            return self._clinical_embedder.encode(query, normalize_embeddings=True).tolist()
        elif self._fast_embedder:
            return self._fast_embedder.encode(query, normalize_embeddings=True).tolist()
        else:
            raise RuntimeError("No embedder available")

    def embed_batch(self, texts: List[str], stage: str = "fast") -> List[List[float]]:
        """Embed a batch of texts."""
        if stage == "clinical" and self._clinical_embedder:
            return self._clinical_embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
        elif self._fast_embedder:
            return self._fast_embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
        else:
            raise RuntimeError("No embedder available")

    def get_cross_encoder(self):
        """Get or initialize the cross-encoder reranker."""
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            except Exception as e:
                logger.warning(f"Could not load cross-encoder: {e}")
        return self._cross_encoder

    def retrieve(
        self,
        query: str,
        vector_store,
        hybrid_search,
        top_k: int = 5,
        use_amer: bool = True,
    ) -> List[Dict]:
        """
        AMER retrieval with multi-stage pipeline.

        Args:
            query: User query
            vector_store: ChromaDB vector store
            hybrid_search: Hybrid search engine
            top_k: Final number of results
            use_amer: Enable AMER multi-stage retrieval

        Returns:
            List of retrieved documents with scores
        """
        if not use_amer or not self._fast_embedder:
            # Fallback to standard hybrid search
            if hybrid_search:
                return self._standard_hybrid(query, hybrid_search, top_k)
            else:
                return self._standard_vector(query, vector_store, top_k)

        # Stage 1: Coarse retrieval with fast embedder
        coarse_results = self._coarse_retrieval(query, vector_store, top_k=20)
        logger.debug(f"AMER Stage 1 (coarse): {len(coarse_results)} results")

        # Stage 2: Mid retrieval with clinical embedder (if available)
        mid_results = []
        if self._clinical_embedder:
            mid_results = self._mid_retrieval(query, vector_store, coarse_results, top_k=15)
            logger.debug(f"AMER Stage 2 (clinical): {len(mid_results)} results")

        # Stage 3: RRF fusion + cross-encoder reranking
        fused = self._rrf_fusion(coarse_results, mid_results, k=60)
        reranked = self._cross_encoder_rerank(query, fused, top_k=top_k)

        logger.debug(f"AMER final: {len(reranked)} results")
        return reranked

    def _coarse_retrieval(self, query: str, vector_store, top_k: int) -> List[Dict]:
        """Stage 1: Fast coarse retrieval."""
        query_emb = self.embed_query(query, stage="fast")
        try:
            results = vector_store.query(query_emb, top_k=top_k)
            docs = []
            for i, doc_id in enumerate(results["ids"][0]):
                docs.append({
                    "id": doc_id,
                    "text": results["documents"][0][i],
                    "score": 1.0 / (1.0 + results["distances"][0][i]),
                    "metadata": results["metadatas"][0][i],
                    "stage": "coarse",
                })
            return docs
        except Exception as e:
            logger.warning(f"Coarse retrieval failed: {e}")
            return []

    def _mid_retrieval(self, query: str, vector_store, coarse_results: List[Dict], top_k: int) -> List[Dict]:
        """Stage 2: Clinical-embedding re-scoring of coarse results."""
        query_emb = self.embed_query(query, stage="clinical")
        try:
            results = vector_store.query(query_emb, top_k=top_k)
            docs = []
            for i, doc_id in enumerate(results["ids"][0]):
                docs.append({
                    "id": doc_id,
                    "text": results["documents"][0][i],
                    "score": 1.0 / (1.0 + results["distances"][0][i]),
                    "metadata": results["metadatas"][0][i],
                    "stage": "clinical",
                })
            return docs
        except Exception as e:
            logger.warning(f"Mid retrieval failed: {e}")
            return []

    def _rrf_fusion(self, list1: List[Dict], list2: List[Dict], k: int = 60) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF) for combining retrieval lists.
        RRF score = sum(1 / (k + rank)) for each list the item appears in.
        """
        rrf_scores = {}
        all_docs = {}

        for rank, doc in enumerate(list1):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
            if doc_id not in all_docs:
                all_docs[doc_id] = doc

        for rank, doc in enumerate(list2):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
            if doc_id not in all_docs:
                all_docs[doc_id] = doc

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)
        fused = []
        for doc_id in sorted_ids:
            doc = dict(all_docs[doc_id])
            doc["rrf_score"] = rrf_scores[doc_id]
            doc["score"] = rrf_scores[doc_id]  # Use RRF as primary score
            fused.append(doc)

        return fused

    def _cross_encoder_rerank(self, query: str, docs: List[Dict], top_k: int) -> List[Dict]:
        """Stage 3: Cross-encoder reranking."""
        ce = self.get_cross_encoder()
        if not ce or len(docs) <= top_k:
            return docs[:top_k]

        try:
            pairs = [(query, d["text"][:512]) for d in docs]
            scores = ce.predict(pairs)
            for i, score in enumerate(scores):
                docs[i]["rerank_score"] = float(score)
            docs.sort(key=lambda d: d.get("rerank_score", 0), reverse=True)
            return docs[:top_k]
        except Exception as e:
            logger.warning(f"Cross-encoder rerank failed: {e}")
            return docs[:top_k]

    def _standard_hybrid(self, query: str, hybrid_search, top_k: int) -> List[Dict]:
        """Fallback: standard hybrid search."""
        try:
            results = hybrid_search.search(query, top_k=top_k)
            docs = []
            for doc_id, fused_score, bm25_score, vec_score, metadata in results:
                docs.append({
                    "id": doc_id, "text": metadata.get("text", ""),
                    "score": fused_score, "metadata": metadata, "stage": "hybrid",
                })
            return docs
        except Exception as e:
            logger.warning(f"Hybrid search failed: {e}")
            return []

    def _standard_vector(self, query: str, vector_store, top_k: int) -> List[Dict]:
        """Fallback: pure vector search."""
        query_emb = self.embed_query(query, stage="fast")
        try:
            results = vector_store.query(query_emb, top_k=top_k)
            docs = []
            for i, doc_id in enumerate(results["ids"][0]):
                docs.append({
                    "id": doc_id, "text": results["documents"][0][i],
                    "score": 1.0 / (1.0 + results["distances"][0][i]),
                    "metadata": results["metadatas"][0][i], "stage": "vector",
                })
            return docs
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []
