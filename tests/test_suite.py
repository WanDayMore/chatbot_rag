import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from starlette.testclient import TestClient
from app.main import app
from app.config import get_settings



def test_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    print("[OK] Health check passed:", data)


def test_cors():
    client = TestClient(app)
    # Preflight OPTIONS request
    response = client.options(
        "/api/chat",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
    print("[OK] CORS preflight check passed.")


def test_rate_limiting():
    client = TestClient(app)
    settings = get_settings()
    limit = settings.RATE_LIMIT_PER_MINUTE  # configured as 3

    # Mock gemini service responses
    from unittest.mock import AsyncMock, patch
    with patch("app.routes.chat.gemini_service.generate_response", new_callable=AsyncMock) as mock_gen, \
         patch("app.cache.semantic_cache.semantic_cache.lookup", return_value=None), \
         patch("app.cache.semantic_cache.semantic_cache.store", return_value=None):
        mock_gen.return_value = "Mocked AI answer."

        print(f"[*] Testing rate limit (threshold: {limit} req/min)...")
        # First 3 requests should pass (200 OK)
        for i in range(limit):
            res = client.post("/api/chat", json={"message": f"ping {i}"})
            assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
            print(f"    - Request {i + 1} passed: {res.json()['reply']}")

        # 4th request MUST hit 429 Too Many Requests
        blocked_res = client.post("/api/chat", json={"message": "ping 4"})
        assert blocked_res.status_code == 429, f"Expected 429, got {blocked_res.status_code}"
        print(f"[OK] Rate limiter successfully blocked 4th request with 429:", blocked_res.json()["detail"])



if __name__ == "__main__":
    test_health()
    test_cors()
    test_rate_limiting()
    print("\nAll automated integration tests PASSED successfully!")

