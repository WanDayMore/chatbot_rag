import asyncio
import logging
from typing import AsyncGenerator, List, Optional
from fastapi import HTTPException, status
from google import genai
from google.genai import types
from app.config import get_settings
from app.llm.prompts import SYSTEM_INSTRUCTION

logger = logging.getLogger("llm.gemini")


class GeminiService:
    def __init__(self):
        self._client: Optional[genai.Client] = None

    def _get_client(self) -> genai.Client:
        if self._client is None:
            settings = get_settings()
            if not settings.GEMINI_API_KEY:
                logger.error("GEMINI_API_KEY is not set in environment or .env file.")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="GEMINI_API_KEY is not configured on the server.",
                )
            self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        return self._client


    def get_embedding(self, text: str) -> List[float]:
        """Generates embedding vector for a single string using text-embedding-004."""
        client = self._get_client()
        settings = get_settings()
        response = client.models.embed_content(
            model=settings.EMBEDDING_MODEL,
            contents=text,
        )
        if response.embeddings and len(response.embeddings) > 0:
            return response.embeddings[0].values
        raise RuntimeError("Failed to generate embedding from Gemini API.")

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a batch of strings."""
        client = self._get_client()
        settings = get_settings()
        response = client.models.embed_content(
            model=settings.EMBEDDING_MODEL,
            contents=texts,
        )
        if response.embeddings:
            return [emb.values for emb in response.embeddings]
        raise RuntimeError("Failed to generate batch embeddings from Gemini API.")

    async def generate_response(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
    ) -> str:
        """Generates a complete response synchronously wrapped in async loop."""
        client = self._get_client()
        settings = get_settings()

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.3,
        )

        # Run synchronous SDK call in thread pool to avoid blocking async event loop
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt,
                config=config,
            ),
        )
        return response.text or ""

    async def stream_response(
        self,
        prompt: str,
    ) -> AsyncGenerator[str, None]:
        """Streams response tokens as an async generator."""
        client = self._get_client()
        settings = get_settings()

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.3,
        )

        def _get_stream():
            return client.models.generate_content_stream(
                model=settings.GEMINI_MODEL,
                contents=prompt,
                config=config,
            )

        loop = asyncio.get_running_loop()
        stream = await loop.run_in_executor(None, _get_stream)

        # Yield chunks
        for chunk in stream:
            if chunk.text:
                yield chunk.text
                await asyncio.sleep(0.01)  # Cooperate with event loop


gemini_service = GeminiService()
