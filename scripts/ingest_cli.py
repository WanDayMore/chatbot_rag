#!/usr/bin/env python3
"""
CLI tool to index local portfolio documents into ChromaDB.

Usage:
    python scripts/ingest_cli.py --dir ./data/knowledge
"""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.ingest import index_documents_from_directory


def main():
    parser = argparse.ArgumentParser(description="Index portfolio knowledge into ChromaDB")
    parser.add_argument(
        "--dir",
        type=str,
        default="./data/knowledge",
        help="Directory containing markdown/txt/json/pdf documents",
    )
    args = parser.parse_args()

    print(f"[*] Scanning and indexing documents in: {args.dir}")
    try:
        result = index_documents_from_directory(args.dir)
        print(f"[OK] Successfully finished indexing!")
        print(f"    - Documents processed: {result['documents_indexed']}")
        print(f"    - Chunks created:     {result['chunks_created']}")
        print(f"    - Status:             {result['status']}")
    except Exception as e:
        print(f"[ERROR] Indexing failed: {e}", file=sys.stderr)
        sys.exit(1)



if __name__ == "__main__":
    main()
