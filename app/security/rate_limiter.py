import time
from datetime import datetime, timezone
from typing import Dict, List
from fastapi import HTTPException, Request, status
from app.config import get_settings


class SecurityRateLimiter:
    """
    In-memory rate limiter enforcing:
    1. Per-IP sliding window limits (minute, hour, day)
    2. Global daily hard cap kill-switch
    """

    def __init__(self):
        # Maps client_ip -> list of timestamps
        self._ip_timestamps: Dict[str, List[float]] = {}
        # Global daily counter: {date_str: count}
        self._global_daily_counts: Dict[str, int] = {}
        self._last_cleanup: float = time.time()

    def _get_client_ip(self, request: Request) -> str:
        # Respect X-Forwarded-For if behind a proxy / CDN like Cloudflare
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        cf_ip = request.headers.get("cf-connecting-ip")
        if cf_ip:
            return cf_ip.strip()
        return request.client.host if request.client else "unknown"

    def _cleanup_old_entries(self, now: float):
        """Periodically cleans up timestamps older than 24 hours to prevent memory leaks."""
        if now - self._last_cleanup < 300:  # Cleanup every 5 minutes
            return

        cutoff = now - 86400
        empty_ips = []
        for ip, timestamps in self._ip_timestamps.items():
            valid_ts = [ts for ts in timestamps if ts > cutoff]
            if valid_ts:
                self._ip_timestamps[ip] = valid_ts
            else:
                empty_ips.append(ip)

        for ip in empty_ips:
            del self._ip_timestamps[ip]

        # Clean old global date keys
        today_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        old_days = [d for d in self._global_daily_counts if d != today_key]
        for d in old_days:
            del self._global_daily_counts[d]

        self._last_cleanup = now

    def check_limits(self, request: Request):
        settings = get_settings()
        now = time.time()
        self._cleanup_old_entries(now)

        # 1. Global Daily Hard Cap Kill-Switch
        today_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        current_global_count = self._global_daily_counts.get(today_key, 0)
        if current_global_count >= settings.DAILY_GLOBAL_HARD_CAP:
            print(
                f"\n[TERMINAL CLUE: DAILY HARD CAP REACHED]\n"
                f"   Total global requests today ({current_global_count}) exceeded DAILY_GLOBAL_HARD_CAP ({settings.DAILY_GLOBAL_HARD_CAP}).\n"
                f"   Client IP: '{self._get_client_ip(request)}' was blocked.\n"
                f"   Action: Reset occurs at UTC midnight, or increase DAILY_GLOBAL_HARD_CAP in .env.\n",
                flush=True
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="The portfolio assistant has reached its maximum daily capacity to keep running costs protected. Please check back tomorrow or contact Isaiah directly!",
            )

        # 2. Per-IP Rate Limits
        client_ip = self._get_client_ip(request)
        if client_ip not in self._ip_timestamps:
            self._ip_timestamps[client_ip] = []

        timestamps = self._ip_timestamps[client_ip]

        # Count requests in last 60 seconds
        minute_cutoff = now - 60
        reqs_in_minute = sum(1 for ts in timestamps if ts > minute_cutoff)
        if reqs_in_minute >= settings.RATE_LIMIT_PER_MINUTE:
            print(
                f"\n[TERMINAL CLUE: MINUTE RATE LIMIT BLOCKED]\n"
                f"   Client IP: '{client_ip}' sent {reqs_in_minute + 1} requests within 60s.\n"
                f"   Limit is: {settings.RATE_LIMIT_PER_MINUTE} req/min.\n"
                f"   Action: Request blocked with 429. User must wait 60s before retrying.\n",
                flush=True
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="You've sent messages a bit too quickly. Please pause for a moment and try again in 1 minute!",
                headers={"Retry-After": "60"},
            )

        # Count requests in last 1 hour (3600 seconds)
        hour_cutoff = now - 3600
        reqs_in_hour = sum(1 for ts in timestamps if ts > hour_cutoff)
        if reqs_in_hour >= settings.RATE_LIMIT_PER_HOUR:
            print(
                f"\n[TERMINAL CLUE: HOURLY LIMIT BLOCKED]\n"
                f"   Client IP: '{client_ip}' reached {reqs_in_hour + 1} requests within 1 hour.\n"
                f"   Limit is: {settings.RATE_LIMIT_PER_HOUR} req/hour.\n"
                f"   Action: Request blocked with 429.\n",
                flush=True
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="You've reached the hourly message limit for this portfolio assistant. Please try again in a little while, or contact Isaiah directly!",
                headers={"Retry-After": "3600"},
            )

        # Count requests in last 24 hours (86400 seconds)
        day_cutoff = now - 86400
        reqs_in_day = sum(1 for ts in timestamps if ts > day_cutoff)
        if reqs_in_day >= settings.RATE_LIMIT_PER_DAY:
            print(
                f"\n[TERMINAL CLUE: DAILY VISITOR QUOTA EXHAUSTED]\n"
                f"   Client IP: '{client_ip}' reached {reqs_in_day + 1} requests in 24 hours.\n"
                f"   Limit is: {settings.RATE_LIMIT_PER_DAY} req/day per visitor.\n"
                f"   Action: Request blocked with 429.\n",
                flush=True
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="You've reached your daily message limit for this portfolio assistant. Feel free to connect with Isaiah directly via the contact form or LinkedIn!",
                headers={"Retry-After": "86400"},
            )

        # Record this request
        timestamps.append(now)
        self._global_daily_counts[today_key] = current_global_count + 1




limiter = SecurityRateLimiter()


def rate_limit_dependency(request: Request):
    """FastAPI Dependency for applying rate limits."""
    limiter.check_limits(request)
