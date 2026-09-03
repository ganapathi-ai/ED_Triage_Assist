"""
Ingestion Script
Processes documents and populates the vector store
"""
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from app.config import settings
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import vector_store
from app.services.embeddings import embedding_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    docs_dir = Path(__file__).resolve().parent.parent.parent / "data"
    logger.info(f"Processing documents from: {docs_dir}")

    processor = DocumentProcessor(str(docs_dir))
    documents = processor.process_all_documents()
    logger.info(f"Processed {len(documents)} documents")

    all_chunks = []
    for doc in documents:
        for chunk in doc.chunks:
            all_chunks.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": {
                    "source": doc.filename,
                    "title": doc.title,
                    "file_type": doc.metadata.get("file_type", "unknown"),
                    "page": chunk.page_number,
                    "section": chunk.section,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                },
            })

    logger.info(f"Total chunks: {len(all_chunks)}")
    if not all_chunks:
        logger.warning("No chunks to ingest")
        return

    vector_store.initialize()
    vector_store.add_documents(all_chunks)

    stats = vector_store.get_stats()
    logger.info(f"Ingestion complete. Vector store: {stats}")


if __name__ == "__main__":
    main()
