import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.config import get_settings
from app.llm.gemini import gemini_service

logger = logging.getLogger("rag.retriever")


class ChromaRetriever:
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
            # Use cosine space for normalized embeddings
            self._collection = self._client.get_or_create_collection(
                name="portfolio_knowledge",
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Retrieves top_k relevant context chunks for the user's query.
        """
        try:
            collection = self._get_collection()
            count = collection.count()
            if count == 0:
                logger.info("ChromaDB portfolio collection is empty. Please run document ingestion.")
                return []

            # Compute embedding with Gemini text-embedding-004
            query_embedding = gemini_service.get_embedding(query)

            # Query ChromaDB
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, count),
                include=["documents", "metadatas", "distances"],
            )

            retrieved_chunks = []
            if results and results["documents"] and len(results["documents"][0]) > 0:
                docs = results["documents"][0]
                metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
                distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

                for doc, meta, dist in zip(docs, metas, distances):
                    # Chroma cosine distance = 1 - cosine_similarity
                    similarity = 1.0 - dist
                    retrieved_chunks.append({
                        "content": doc,
                        "source": meta.get("source", "portfolio_docs") if meta else "portfolio_docs",
                        "title": meta.get("title", "") if meta else "",
                        "similarity": round(similarity, 4),
                    })

                # Sort descending by similarity
                retrieved_chunks.sort(key=lambda c: c["similarity"], reverse=True)

            return retrieved_chunks

        except Exception as e:
            logger.error(f"Error during RAG retrieval: {e}", exc_info=True)
            return []

    def retrieve_with_confidence(self, query: str, top_k: int = 4, min_similarity: float = 0.35) -> tuple[List[Dict[str, Any]], bool]:
        """
        Retrieves top_k chunks and returns (filtered_chunks, has_high_confidence).
        Filters out low-similarity noise when higher confidence chunks exist.
        """
        all_chunks = self.retrieve(query, top_k=top_k)
        if not all_chunks:
            return [], False

        max_similarity = all_chunks[0]["similarity"]
        has_high_confidence = max_similarity >= 0.45

        # If highest similarity is decent, filter out noisy chunks below min_similarity
        if has_high_confidence:
            meaningful_chunks = [c for c in all_chunks if c["similarity"] >= min_similarity]
            return (meaningful_chunks if meaningful_chunks else all_chunks[:1]), True
        else:
            # Low confidence: Keep top 2 chunks as weak background, but flag has_high_confidence = False
            return all_chunks[:2], False



retriever = ChromaRetriever()
