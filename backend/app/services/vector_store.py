"""
ChromaDB Vector Store
Handles document storage, retrieval, and collection management
"""
import logging
import os
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.config import settings
from app.services.embeddings import embedding_service

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB-based vector store for document embeddings."""

    def __init__(self):
        self.persist_dir = settings.chroma_persist_dir
        self.collection_name = settings.chroma_collection_name
        self.client: Optional[chromadb.Client] = None
        self.collection = None

    def initialize(self):
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Vector store initialized: {self.collection_name}")

    def add_documents(self, chunks: List[dict], embeddings: List[List[float]] = None):
        if not self.collection:
            self.initialize()
        if not chunks:
            return
        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [c.get("metadata", {}) for c in chunks]
        if embeddings is None:
            embeddings = embedding_service.embed_documents(documents)
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i:i+batch_size],
                documents=documents[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
            )
        logger.info(f"Added {len(chunks)} chunks to vector store")

    def query(self, query_embedding: List[float], top_k: int = 20) -> Dict:
        if not self.collection:
            self.initialize()
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return results

    def get_all_documents(self) -> List[Dict]:
        if not self.collection:
            self.initialize()
        results = self.collection.get(include=["documents", "metadatas"])
        docs = []
        for i in range(len(results["ids"])):
            docs.append({"id": results["ids"][i], "text": results["documents"][i], "metadata": results["metadatas"][i]})
        return docs

    def get_document_count(self) -> int:
        if not self.collection:
            self.initialize()
        return self.collection.count()

    def delete_collection(self):
        if self.client:
            try:
                self.client.delete_collection(self.collection_name)
                logger.info(f"Deleted collection: {self.collection_name}")
            except Exception:
                pass

    def get_stats(self) -> Dict:
        if not self.collection:
            self.initialize()
        return {"collection_name": self.collection_name, "document_count": self.collection.count()}


vector_store = VectorStore()
