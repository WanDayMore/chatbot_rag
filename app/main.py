import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.models.schemas import IngestResponse
from app.rag.ingest import index_documents_from_directory
from app.routes.chat import router as chat_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Initializing Portfolio RAG Chatbot Service...")
    logger.info(f"Allowed CORS Origins: {settings.cors_origins}")
    logger.info(f"Cloudflare Turnstile Active: {settings.TURNSTILE_ENABLED}")
    logger.info(f"Gemini Model: {settings.GEMINI_MODEL}")
    yield
    logger.info("Shutting down Portfolio RAG Chatbot Service.")


app = FastAPI(
    title="Portfolio RAG Chatbot API",
    description="Secured, rate-limited, semantically-cached RAG service powered by Gemini Pro",
    version="1.0.0",
    lifespan=lifespan,
)

settings = get_settings()

# Strict CORS Whitelist
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Register routes
app.include_router(chat_router)


@app.get("/health", tags=["system"])
async def health_check():
    """Health check endpoint for container uptime monitoring."""
    return {
        "status": "healthy",
        "turnstile_enabled": settings.TURNSTILE_ENABLED,
        "model": settings.GEMINI_MODEL,
    }


@app.post("/api/admin/ingest", response_model=IngestResponse, tags=["admin"])
async def trigger_ingest(
    folder_path: str = "./data/knowledge",
    admin_key: str = Header(None, alias="X-Admin-Key"),
):
    """
    Re-indexes documents from a server directory into ChromaDB.
    Protected by Gemini API key or custom key.
    """
    if admin_key != settings.GEMINI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing admin key.",
        )

    result = index_documents_from_directory(folder_path)
    return IngestResponse(**result)
