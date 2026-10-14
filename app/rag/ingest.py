import hashlib
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pypdf
from chromadb.config import Settings as ChromaSettings
import chromadb
from app.config import get_settings
from app.llm.gemini import gemini_service

logger = logging.getLogger("rag.ingest")


def chunk_text(text: str, chunk_size: int = 700, chunk_overlap: int = 100) -> List[str]:
    """
    Splits text into chunks by paragraphs/newlines, preserving context overlap.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk = f"{current_chunk}\n\n{para}".strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # If paragraph itself is longer than chunk_size, split by sentences/lines
            if len(para) > chunk_size:
                lines = para.split("\n")
                sub_chunk = ""
                for line in lines:
                    if len(sub_chunk) + len(line) <= chunk_size:
                        sub_chunk = f"{sub_chunk}\n{line}".strip()
                    else:
                        if sub_chunk:
                            chunks.append(sub_chunk)
                        sub_chunk = line
                if sub_chunk:
                    current_chunk = sub_chunk
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def load_file_content(file_path: Path) -> str:
    """Extracts text from Markdown, Text, JSON, or PDF files."""
    suffix = file_path.suffix.lower()

    if suffix in [".md", ".txt"]:
        return file_path.read_text(encoding="utf-8", errors="ignore")

    elif suffix == ".json":
        data = json.loads(file_path.read_text(encoding="utf-8", errors="ignore"))
        return json.dumps(data, indent=2)

    elif suffix == ".pdf":
        text_parts = []
        reader = pypdf.PdfReader(str(file_path))
        for page_num, page in enumerate(reader.pages):
            extracted = page.extract_text()
            if extracted:
                text_parts.append(f"--- Page {page_num + 1} ---\n{extracted}")
        return "\n\n".join(text_parts)

    else:
        logger.warning(f"Unsupported file format: {file_path.name}")
        return ""


def index_documents_from_directory(dir_path: str | Path) -> Dict[str, Any]:
    """
    Scans a directory for portfolio knowledge documents and indexes them in ChromaDB.
    """
    settings = get_settings()
    path = Path(dir_path)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    files = [
        f for f in path.rglob("*")
        if f.is_file() and f.suffix.lower() in [".md", ".txt", ".json", ".pdf"]
    ]

    if not files:
        logger.warning(f"No indexable files found in {dir_path}")
        return {"documents_indexed": 0, "chunks_created": 0, "status": "no files found"}

    all_chunks: List[str] = []
    all_metas: List[Dict[str, Any]] = []
    all_ids: List[str] = []

    for file_path in files:
        raw_text = load_file_content(file_path)
        if not raw_text.strip():
            continue

        chunks = chunk_text(raw_text)
        for idx, chunk in enumerate(chunks):
            chunk_id = hashlib.sha256(f"{file_path.name}_{idx}_{chunk[:40]}".encode("utf-8")).hexdigest()[:16]
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metas.append({
                "source": file_path.name,
                "title": file_path.stem.replace("_", " ").title(),
                "chunk_index": idx,
            })

    if not all_chunks:
        return {"documents_indexed": len(files), "chunks_created": 0, "status": "empty contents"}

    # Generate embeddings in batches of 10
    logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
    batch_size = 10
    all_embeddings = []
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        embeddings = gemini_service.get_embeddings_batch(batch)
        all_embeddings.extend(embeddings)

    # Store into ChromaDB
    client = chromadb.PersistentClient(
        path=settings.CHROMA_PERSIST_DIRECTORY,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name="portfolio_knowledge",
        metadata={"hnsw:space": "cosine"},
    )

    # Upsert to avoid duplicate IDs on re-runs
    collection.upsert(
        ids=all_ids,
        embeddings=all_embeddings,
        documents=all_chunks,
        metadatas=all_metas,
    )

    logger.info(f"Successfully indexed {len(files)} files into {len(all_chunks)} chunks.")
    return {
        "documents_indexed": len(files),
        "chunks_created": len(all_chunks),
        "status": "success",
    }
