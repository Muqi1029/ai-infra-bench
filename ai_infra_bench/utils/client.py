import sys

import aiohttp

BENCH_AIOHTTP_TIMEOUT_SECONDS = 6 * 60 * 60  # 6 hours
BENCH_AIOHTTP_READ_BUFSIZE_BYTES = 10 * 1024**2  # 10 MB


def _create_bench_client_session(
    max_concurrency: int | None = 32,
    api_key: str | None = "EMPTY",
    request_timeout: float = BENCH_AIOHTTP_TIMEOUT_SECONDS,
) -> aiohttp.ClientSession:
    # In open-loop QPS mode (max_concurrency=None), pass 0 to disable aiohttp's
    # default 100-connection ceiling so arrivals are never client-throttled.
    conn_limit = 0 if max_concurrency is None else max_concurrency
    aiohttp_timeout = aiohttp.ClientTimeout(total=request_timeout)
    connector_kwargs = {
        "limit": conn_limit,
        "limit_per_host": conn_limit,
        # Expire pooled connections before SGLang's default 5s idle timeout.
        # Large payload preparation can block the loop from observing a peer's
        # close; aiohttp also checks this age synchronously before reuse.
        "keepalive_timeout": 1.0,
    }
    if sys.version_info < (3, 12, 7):
        connector_kwargs["enable_cleanup_closed"] = True
    connector = aiohttp.TCPConnector(**connector_kwargs)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    return aiohttp.ClientSession(
        timeout=aiohttp_timeout,
        read_bufsize=BENCH_AIOHTTP_READ_BUFSIZE_BYTES,
        connector=connector,
        headers=headers,
    )
