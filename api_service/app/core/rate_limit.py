"""
Rate limiting middleware for API endpoints.
"""
from fastapi import Request, Response, status
import time
import logging
from typing import Dict, Tuple, Callable, Any, Optional
from dataclasses import dataclass
import asyncio

from app.core.config import settings
from app.core.exceptions import RateLimitError
from app.core.logging import logger


@dataclass
class RateLimitRule:
    """Rate limit rule configuration."""
    requests: int  # Number of requests allowed
    period: int    # Time period in seconds
    burst: int     # Burst capacity


class RateLimiter:
    """
    Token bucket rate limiter implementation.
    """
    
    def __init__(self):
        """
        Initialize the rate limiter.
        """
        self._buckets: Dict[str, Tuple[float, float]] = {}
        self._lock = asyncio.Lock()
        
        # Default rule from settings
        self.default_rule = RateLimitRule(
            requests=settings.RATE_LIMIT_DEFAULT,
            period=60,  # 1 minute
            burst=settings.RATE_LIMIT_BURST
        )
        
        # Endpoint-specific rules (can be customized)
        self.endpoint_rules: Dict[str, RateLimitRule] = {
            # Example: Higher limits for critical endpoints
            "/api/v1/health": RateLimitRule(requests=300, period=60, burst=350),
            # Example: Lower limits for expensive endpoints
            "/api/v1/analytics/export": RateLimitRule(requests=10, period=60, burst=15),
        }
    
    async def _get_bucket(self, key: str) -> Tuple[float, float]:
        """
        Get or create a token bucket for the given key.
        
        Args:
            key: The rate limit key (typically IP + endpoint)
            
        Returns:
            Tuple of (tokens, last_refill)
        """
        async with self._lock:
            if key not in self._buckets:
                # Create a new bucket with full tokens
                rule = self._get_rule_for_key(key)
                self._buckets[key] = (rule.burst, time.time())
            
            return self._buckets[key]
    
    async def _update_bucket(self, key: str, tokens: float, last_refill: float) -> None:
        """
        Update a token bucket.
        
        Args:
            key: The rate limit key
            tokens: Current token count
            last_refill: Last refill timestamp
        """
        async with self._lock:
            self._buckets[key] = (tokens, last_refill)
            
            # Clean up old buckets occasionally to prevent memory leaks
            # This is a simple approach; a more robust solution would use a separate cleanup task
            if len(self._buckets) > 10000:
                current_time = time.time()
                keys_to_remove = [
                    k for k, (_, last_refill) in self._buckets.items()
                    if current_time - last_refill > 3600  # 1 hour
                ]
                for k in keys_to_remove:
                    self._buckets.pop(k, None)
    
    def _get_rule_for_key(self, key: str) -> RateLimitRule:
        """
        Get the rate limit rule for a given key.
        
        Args:
            key: The rate limit key
            
        Returns:
            RateLimitRule configuration
        """
        # Extract endpoint path from key
        parts = key.split("|")
        if len(parts) > 1:
            endpoint = parts[1]
            # Check if we have a specific rule for this endpoint
            for rule_endpoint, rule in self.endpoint_rules.items():
                if endpoint.startswith(rule_endpoint):
                    return rule
        
        # Return default rule
        return self.default_rule
    
    async def check_rate_limit(self, key: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a request is allowed based on rate limits.
        
        Args:
            key: The rate limit key
            
        Returns:
            Tuple of (allowed, limit_info)
        """
        if not settings.RATE_LIMIT_ENABLED:
            # Rate limiting disabled
            return True, {}
        
        tokens, last_refill = await self._get_bucket(key)
        
        # Get rule for this key
        rule = self._get_rule_for_key(key)
        
        # Calculate token refill
        now = time.time()
        time_passed = now - last_refill
        refill_amount = time_passed * (rule.requests / rule.period)
        
        # Update tokens
        new_tokens = min(rule.burst, tokens + refill_amount)
        
        # Check if request can be allowed
        allowed = new_tokens >= 1.0
        if allowed:
            # Consume a token
            new_tokens -= 1.0
        
        # Update bucket
        await self._update_bucket(key, new_tokens, now)
        
        # Prepare rate limit info headers
        reset_seconds = 0
        if new_tokens < rule.requests:
            # Calculate time until full refill
            tokens_needed = rule.requests - new_tokens
            reset_seconds = int(tokens_needed * rule.period / rule.requests)
        
        limit_info = {
            "limit": rule.requests,
            "remaining": int(new_tokens),
            "reset": reset_seconds,
            "allowed": allowed
        }
        
        return allowed, limit_info


# Singleton instance
rate_limiter = RateLimiter()


async def rate_limit_middleware(request: Request, call_next: Callable) -> Response:
    """
    Rate limiting middleware for FastAPI.
    
    Args:
        request: FastAPI request
        call_next: Next middleware function
        
    Returns:
        FastAPI response
        
    Raises:
        RateLimitError: If request is rate limited
    """
    if not settings.RATE_LIMIT_ENABLED:
        # Rate limiting disabled, skip to next middleware
        return await call_next(request)
    
    # Create rate limit key based on IP and endpoint
    ip = request.client.host if request.client else "unknown"
    endpoint = request.url.path
    key = f"{ip}|{endpoint}"
    
    # Check rate limit
    allowed, limit_info = await rate_limiter.check_rate_limit(key)
    
    if not allowed:
        # Log rate limit exceeded
        logger.warning(f"Rate limit exceeded: {key}")
        
        # Return rate limit error
        raise RateLimitError("Rate limit exceeded")
    
    # Process request
    response = await call_next(request)
    
    # Add rate limit headers
    response.headers["X-RateLimit-Limit"] = str(limit_info["limit"])
    response.headers["X-RateLimit-Remaining"] = str(limit_info["remaining"])
    response.headers["X-RateLimit-Reset"] = str(limit_info["reset"])
    
    return response
