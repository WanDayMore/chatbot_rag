import logging
import httpx
from fastapi import HTTPException, status
from app.config import get_settings

logger = logging.getLogger("security.turnstile")

CLOUDFLARE_TURNSTILE_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"


async def verify_turnstile_token(token: str | None, client_ip: str | None = None) -> bool:
    """
    Verifies Cloudflare Turnstile token.
    If TURNSTILE_ENABLED is False (e.g. during development), validation succeeds automatically.
    """
    settings = get_settings()

    if not settings.TURNSTILE_ENABLED:
        logger.debug("Turnstile verification disabled (development mode).")
        return True

    if not settings.TURNSTILE_SECRET_KEY:
        logger.warning("Turnstile enabled but TURNSTILE_SECRET_KEY is empty. Rejecting request.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server security configuration error.",
        )

    if not token:
        logger.warning(f"Missing Turnstile token from IP: {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Security verification required. Missing Turnstile token.",
        )

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            payload = {
                "secret": settings.TURNSTILE_SECRET_KEY,
                "response": token,
            }
            if client_ip:
                payload["remoteip"] = client_ip

            response = await client.post(CLOUDFLARE_TURNSTILE_VERIFY_URL, data=payload)
            result = response.json()

            if result.get("success", False):
                return True
            else:
                error_codes = result.get("error-codes", [])
                logger.warning(f"Turnstile verification failed for IP {client_ip}: {error_codes}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Bot verification challenge failed.",
                )
    except httpx.RequestError as exc:
        logger.error(f"Error contacting Cloudflare Turnstile: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Verification service temporarily unavailable.",
        )
