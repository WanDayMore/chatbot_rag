import hashlib
import logging
import time
from typing import Optional, Tuple, Dict, Any
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.config import get_settings
from app.llm.gemini import gemini_service

logger = logging.getLogger("cache.semantic")


class SemanticQueryCache:
    """
    Semantic similarity cache using ChromaDB.
    Matches queries by vector cosine similarity to eliminate redundant LLM calls.
    """

    def __init__(self):
        self._client: Optional[chromadb.ClientAPI] = None
        self._collection = None

    def _get_collection(self):
        if self._collection is None:
            settings = get_settings()
            self._client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIRECTORY,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name="semantic_query_cache",
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def lookup(self, query: str) -> Optional[Tuple[str, float]]:
        """
        Looks up a query in semantic cache.
        Returns (cached_answer, similarity) if similarity >= threshold, else None.
        """
        settings = get_settings()
        collection = self._get_collection()

        count = collection.count()
        if count == 0:
            return None

        try:
            query_embedding = gemini_service.get_embedding(query)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=1,
                include=["metadatas", "distances"],
            )

            if results and results["distances"] and len(results["distances"][0]) > 0:
                dist = results["distances"][0][0]
                similarity = 1.0 - dist
                metadata = results["metadatas"][0][0]

                if similarity >= settings.SEMANTIC_CACHE_SIMILARITY_THRESHOLD:
                    # Check TTL
                    cached_time = metadata.get("timestamp", 0)
                    ttl_seconds = settings.SEMANTIC_CACHE_TTL_DAYS * 86400
                    if time.time() - cached_time < ttl_seconds:
                        cached_answer = metadata.get("answer", "")
                        logger.info(
                            f"Semantic cache HIT (similarity: {similarity:.4f} >= {settings.SEMANTIC_CACHE_SIMILARITY_THRESHOLD}) for query: '{query[:50]}...'"
                        )
                        return cached_answer, similarity
                    else:
                        logger.debug("Cache entry expired by TTL.")
                else:
                    logger.debug(
                        f"Semantic cache MISS (closest similarity: {similarity:.4f} < {settings.SEMANTIC_CACHE_SIMILARITY_THRESHOLD})"
                    )

            return None
        except Exception as e:
            logger.error(f"Error checking semantic cache: {e}", exc_info=True)
            return None

    def store(self, query: str, answer: str, query_embedding: Optional[list] = None):
        """
        Stores an answered query into the semantic cache.
        """
        try:
            collection = self._get_collection()
            if query_embedding is None:
                query_embedding = gemini_service.get_embedding(query)

            cache_id = hashlib.sha256(f"{query}_{time.time()}".encode("utf-8")).hexdigest()[:16]

            collection.add(
                ids=[cache_id],
                embeddings=[query_embedding],
                documents=[query],
                metadatas=[{
                    "query": query,
                    "answer": answer,
                    "timestamp": time.time(),
                }],
            )
            logger.info(f"Stored query response in semantic cache (id={cache_id}).")
        except Exception as e:
            logger.error(f"Error storing in semantic cache: {e}", exc_info=True)


semantic_cache = SemanticQueryCache()
