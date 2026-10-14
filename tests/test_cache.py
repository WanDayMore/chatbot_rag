import time
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.cache.semantic_cache import SemanticQueryCache


def test_cache():
    cache = SemanticQueryCache()
    # Mock embedding generator with deterministic vectors matching gemini-embedding-001 (3072 dims)
    mock_emb = [0.1] * 3072


    cache_collection = cache._get_collection()
    
    # Clean previous test entries if any
    try:
        existing = cache_collection.get()
        if existing and existing["ids"]:
            cache_collection.delete(ids=existing["ids"])
    except Exception:
        pass

    query = "What programming languages does Isaiah know?"
    answer = "Isaiah is proficient in Python, TypeScript, SQL, and modern web frameworks."

    # 1. Lookup on empty cache -> None
    from unittest.mock import patch
    with patch("app.cache.semantic_cache.gemini_service.get_embedding", return_value=mock_emb):
        res = cache.lookup(query)
        assert res is None, "Expected cache miss on empty cache"
        print("[OK] Empty cache miss confirmed.")

        # 2. Store in cache
        t0 = time.perf_counter()
        cache.store(query, answer, query_embedding=mock_emb)
        print(f"[OK] Stored entry in cache.")

        # 3. Lookup exact query -> HIT
        hit = cache.lookup(query)
        assert hit is not None, "Expected cache hit"
        cached_ans, sim = hit
        assert cached_ans == answer
        assert sim >= 0.99
        print(f"[OK] Cache HIT verified! Similarity: {sim:.4f}, Answer: '{cached_ans}'")

if __name__ == "__main__":
    test_cache()
    print("\nAll Semantic Cache tests PASSED successfully!")
