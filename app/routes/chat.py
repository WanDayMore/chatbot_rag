import json
import logging
import time
from fastapi import APIRouter, Depends, Request
from sse_starlette.sse import EventSourceResponse

from app.cache.semantic_cache import semantic_cache
from app.config import get_settings
from app.llm.gemini import gemini_service
from app.llm.prompts import construct_rag_prompt
from app.models.schemas import ChatRequest, ChatResponse, SourceDocument
from app.rag.retriever import retriever
from app.security.rate_limiter import rate_limit_dependency
from app.security.turnstile import verify_turnstile_token

logger = logging.getLogger("routes.chat")
router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse, dependencies=[Depends(rate_limit_dependency)])
async def chat_json(request: Request, body: ChatRequest):
    """
    Standard non-streaming JSON endpoint.
    Checks semantic cache first; on miss, retrieves RAG context and calls Gemini Pro.
    """
    settings = get_settings()
    start_time = time.perf_counter()


    # 1. Verify Turnstile token if enabled
    client_ip = request.client.host if request.client else None
    await verify_turnstile_token(body.turnstile_token, client_ip=client_ip)

    # 2. Check Semantic Cache
    cached_result = semantic_cache.lookup(body.message)
    if cached_result:
        cached_answer, similarity = cached_result
        latency_ms = (time.perf_counter() - start_time) * 1000
        return ChatResponse(
            reply=cached_answer,
            cached=True,
            sources=[],
            latency_ms=round(latency_ms, 2),
        )

    # 3. Cache Miss: Retrieve relevant knowledge from ChromaDB
    retrieved, has_high_confidence = retriever.retrieve_with_confidence(body.message, top_k=4)
    if not has_high_confidence:
        top_score = retrieved[0]["similarity"] if retrieved else 0.0
        print(
            f"\n[TERMINAL CLUE: LOW RELEVANCE / UNFAMILIAR TERM DETECTED]\n"
            f"   Question: '{body.message}'\n"
            f"   Highest Similarity Score: {top_score:.4f} (Threshold: 0.45)\n"
            f"   Diagnosis: The query may contain typos or refer to terms/tools not present in Isaiah's portfolio.\n"
            f"   Action: Injected advisory into prompt to prevent hallucination and suggest related skills.\n",
            flush=True
        )

    context_text = "\n\n".join([
        f"Document: {r['source']}\n{r['content']}" for r in retrieved
    ])

    # 4. Generate response via Gemini
    prompt = construct_rag_prompt(body.message, context_text, has_high_confidence=has_high_confidence)

    try:
        reply = await gemini_service.generate_response(prompt)
    except Exception as exc:
        print(
            f"\n[TERMINAL CLUE: GEMINI GENERATION FAILED]\n"
            f"   Model: {settings.GEMINI_MODEL}\n"
            f"   Visitor Question: '{body.message}'\n"
            f"   Upstream Exception: {exc}\n"
            f"   Why response is altered: Upstream AI API encountered an error (e.g. temporary 503 high demand or quota).\n"
            f"   UI Status: Friendly notice served without exposing stack trace.\n",
            flush=True
        )
        reply = "I'm temporarily experiencing connectivity issues with the AI service. Please try asking again in a moment, or reach out to Isaiah directly via email!"

    # Guard against blank response
    if not reply or not reply.strip():
        print(
            f"\n[TERMINAL CLUE: BLANK RESPONSE DETECTED]\n"
            f"   Model: {settings.GEMINI_MODEL} returned empty/blank text.\n"
            f"   Visitor Question: '{body.message}'\n"
            f"   Why response is blank: Possible Google safety filter block, high-demand timeout, or empty model output.\n"
            f"   UI Status: Friendly notice served.\n",
            flush=True
        )

        reply = "I apologize, but I couldn't formulate a response for this query. Please feel free to ask in a different way or contact Isaiah directly!"
    else:
        # 5. Populate cache only on valid non-blank response
        semantic_cache.store(body.message, reply)

    latency_ms = (time.perf_counter() - start_time) * 1000
    # Only cite sources with meaningful relevance (score >= 0.40) to prevent misleading citations on off-topic questions
    sources = [
        SourceDocument(
            source=r["source"],
            content_snippet=r["content"][:200] + "..." if len(r["content"]) > 200 else r["content"],
            score=r["similarity"],
        )
        for r in retrieved
        if r.get("similarity", 0.0) >= 0.40
    ]

    return ChatResponse(
        reply=reply,
        cached=False,
        sources=sources,
        latency_ms=round(latency_ms, 2),
    )


@router.post("/stream", dependencies=[Depends(rate_limit_dependency)])
async def chat_stream(request: Request, body: ChatRequest):
    """
    Server-Sent Events (SSE) streaming endpoint.
    Emits data chunks for a typewriter effect, followed by [DONE].
    Never exposes stack traces to UI; prints diagnostic clues in terminal.
    """
    settings = get_settings()

    # 1. Verify Turnstile
    client_ip = request.client.host if request.client else None
    await verify_turnstile_token(body.turnstile_token, client_ip=client_ip)

    async def event_generator():
        # 2. Check Semantic Cache
        cached_result = semantic_cache.lookup(body.message)
        if cached_result:
            cached_answer, _ = cached_result
            yield {
                "event": "message",
                "data": json.dumps({"chunk": cached_answer, "cached": True}),
            }
            yield {
                "event": "end",
                "data": json.dumps({"done": True, "cached": True}),
            }
            return

        # 3. Retrieve context with confidence scoring
        retrieved, has_high_confidence = retriever.retrieve_with_confidence(body.message, top_k=4)
        if not has_high_confidence:
            top_score = retrieved[0]["similarity"] if retrieved else 0.0
            print(
                f"\n[TERMINAL CLUE: LOW RELEVANCE / UNFAMILIAR TERM DETECTED (STREAM)]\n"
                f"   Question: '{body.message}'\n"
                f"   Highest Similarity Score: {top_score:.4f} (Threshold: 0.45)\n"
                f"   Diagnosis: The query may contain typos or refer to terms/tools not present in Isaiah's portfolio.\n"
                f"   Action: Injected advisory into prompt to prevent hallucination and suggest related skills.\n",
                flush=True
            )

        context_text = "\n\n".join([
            f"Document: {r['source']}\n{r['content']}" for r in retrieved
        ])

        # 4. Stream response
        prompt = construct_rag_prompt(body.message, context_text, has_high_confidence=has_high_confidence)
        full_reply_parts = []

        # Only emit sources metadata for relevant documents (score >= 0.40)
        sources = [
            {"source": r["source"], "title": r.get("title", ""), "score": r["similarity"]}
            for r in retrieved
            if r.get("similarity", 0.0) >= 0.40
        ]
        yield {
            "event": "sources",
            "data": json.dumps({"sources": sources}),
        }


        try:
            async for chunk in gemini_service.stream_response(prompt):
                full_reply_parts.append(chunk)
                yield {
                    "event": "message",
                    "data": json.dumps({"chunk": chunk, "cached": False}),
                }
        except Exception as exc:
            print(
                f"\n[TERMINAL CLUE: STREAM GENERATION INTERRUPTED]\n"
                f"   Model: {settings.GEMINI_MODEL}\n"
                f"   Visitor Question: '{body.message}'\n"
                f"   Exception: {exc}\n"
                f"   Why stream stopped: Upstream Gemini stream disconnected or failed.\n",
                flush=True
            )
            fallback_msg = " ... [Connection temporarily interrupted. Please try again shortly or reach out to Isaiah directly!]"
            yield {
                "event": "message",
                "data": json.dumps({"chunk": fallback_msg, "cached": False}),
            }

        # Check if full reply ended up blank
        full_reply = "".join(full_reply_parts).strip()
        if not full_reply:
            print(
                f"\n[TERMINAL CLUE: STREAM RESULT WAS BLANK]\n"
                f"   Model: {settings.GEMINI_MODEL} generated 0 streaming tokens.\n"
                f"   Visitor Question: '{body.message}'\n"
                f"   Why blank: Upstream model yielded no content.\n",
                flush=True
            )

            yield {
                "event": "message",
                "data": json.dumps({
                    "chunk": "I apologize, but I wasn't able to produce a response to that question. Please feel free to ask something else!",
                    "cached": False
                }),
            }
        else:
            # Save valid full response to cache
            semantic_cache.store(body.message, full_reply)

        yield {
            "event": "end",
            "data": json.dumps({"done": True, "cached": False}),
        }

    return EventSourceResponse(event_generator())

