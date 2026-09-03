"""
Advanced Chunking Strategy
Implements Parent-Child chunking for better retrieval granularity
"""
from typing import List, Tuple
from dataclasses import dataclass
import re


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    chunk_id: str
    parent_id: Optional[str]
    metadata: dict
    chunk_index: int


class ParentChildChunker:
    """
    Parent-Child chunking strategy:
    - Parent chunks: Large context windows (1024 tokens) for LLM consumption
    - Child chunks: Small granular chunks (256 tokens) for precise retrieval
    - Each child points to its parent for full context retrieval
    """

    def __init__(
        self,
        parent_size: int = 1024,
        parent_overlap: int = 100,
        child_size: int = 256,
        child_overlap: int = 30,
    ):
        self.parent_size = parent_size
        self.parent_overlap = parent_overlap
        self.child_size = child_size
        self.child_overlap = child_overlap

    def chunk_document(self, text: str, doc_id: str, metadata: dict = None) -> Tuple[List[Chunk], List[Chunk]]:
        """
        Create parent-child chunks from a document.
        Returns (parent_chunks, child_chunks).
        """
        # First create parent chunks
        parent_chunks = self._create_parent_chunks(text, doc_id, metadata)
        # Then create child chunks from each parent
        child_chunks = self._create_child_chunks(parent_chunks)

        return parent_chunks, child_chunks

    def _create_parent_chunks(self, text: str, doc_id: str, metadata: dict) -> List[Chunk]:
        """Create parent-level chunks."""
        chunks = []
        words = text.split()
        start = 0
        chunk_idx = 0

        while start < len(words):
            end = start + self.parent_size
            chunk_text = " ".join(words[start:end])

            chunk = Chunk(
                text=chunk_text,
                chunk_id=f"{doc_id}_p{chunk_idx}",
                parent_id=None,  # Parents don't have parents
                metadata={**(metadata or {}), "chunk_type": "parent", "doc_id": doc_id},
                chunk_index=chunk_idx,
            )
            chunks.append(chunk)
            chunk_idx += 1
            start = end - self.parent_overlap

        return chunks

    def _create_child_chunks(self, parent_chunks: List[Chunk]) -> List[Chunk]:
        """Create child chunks from each parent chunk."""
        child_chunks = []
        child_idx = 0

        for parent in parent_chunks:
            words = parent.text.split()
            start = 0

            while start < len(words):
                end = start + self.child_size
                child_text = " ".join(words[start:end])

                child = Chunk(
                    text=child_text,
                    chunk_id=f"{parent.chunk_id}_c{child_idx}",
                    parent_id=parent.chunk_id,
                    metadata={
                        **parent.metadata,
                        "chunk_type": "child",
                        "parent_chunk_id": parent.chunk_id,
                    },
                    chunk_index=child_idx,
                )
                child_chunks.append(child)
                child_idx += 1
                start = end - self.child_overlap

        return child_chunks

    def get_parent_context(self, child_chunk: Chunk, all_parents: List[Chunk]) -> str:
        """Get the full parent context for a child chunk."""
        for parent in all_parents:
            if parent.chunk_id == child_chunk.parent_id:
                return parent.text
        return child_chunk.text


class SemanticChunker:
    """
    Semantic chunking based on content boundaries.
    Splits at paragraph breaks, section headers, and topic boundaries.
    """

    def __init__(self, target_size: int = 400, max_size: int = 800):
        self.target_size = target_size
        self.max_size = max_size

    def chunk_text(self, text: str, doc_id: str, metadata: dict = None) -> List[Chunk]:
        """Chunk text at semantic boundaries."""
        chunks = []

        # Split into sections (by headers and major breaks)
        sections = self._split_into_sections(text)

        chunk_idx = 0
        for section_text, section_name in sections:
            # Split section into sub-chunks if too large
            sub_chunks = self._split_section(section_text)

            for i, sub_text in enumerate(sub_chunks):
                if len(sub_text.split()) < 20:  # Skip very short chunks
                    continue

                chunk = Chunk(
                    text=sub_text,
                    chunk_id=f"{doc_id}_s{chunk_idx}",
                    parent_id=None,
                    metadata={
                        **(metadata or {}),
                        "section": section_name,
                        "chunk_type": "semantic",
                        "word_count": len(sub_text.split()),
                    },
                    chunk_index=chunk_idx,
                )
                chunks.append(chunk)
                chunk_idx += 1

        return chunks

    def _split_into_sections(self, text: str) -> List[Tuple[str, str]]:
        """Split text into sections based on headers."""
        sections = []
        lines = text.split('\n')

        current_section = "Introduction"
        current_content = []

        for line in lines:
            # Detect section headers
            if re.match(r'^#{1,3}\s+', line) or \
               re.match(r'^[A-Z][A-Z\s]{5,}$', line.strip()) or \
               re.match(r'^\d+\.\s+[A-Z]', line):
                if current_content:
                    sections.append(("\n".join(current_content), current_section))
                current_section = line.strip().lstrip('#').strip()
                current_content = []
            else:
                current_content.append(line)

        if current_content:
            sections.append(("\n".join(current_content), current_section))

        return sections if sections else [(text, "Full Document")]

    def _split_section(self, text: str) -> List[str]:
        """Split a section into sub-chunks at paragraph boundaries."""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_words = len(para.split())

            if current_size + para_words > self.max_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_size = 0

            current_chunk.append(para)
            current_size += para_words

            if current_size >= self.target_size:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_size = 0

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks
