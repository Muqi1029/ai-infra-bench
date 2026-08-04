import aiohttp

BENCH_AIOHTTP_TIMEOUT_SECONDS = 6 * 60 * 60  # 6 hours
BENCH_AIOHTTP_READ_BUFSIZE_BYTES = 10 * 1024**2  # 10 MB


def _create_bench_client_session(
    max_concurrency: int = 32, api_key: str = "EMPTY"
) -> aiohttp.ClientSession:
    # When the pressure is big, the read buffer could be full before aio thread read
    # the content. We increase the read_bufsize from 64K to 10M.
    aiohttp_timeout = aiohttp.ClientTimeout(total=BENCH_AIOHTTP_TIMEOUT_SECONDS)
    connector = aiohttp.TCPConnector(
        limit=max_concurrency,
        limit_per_host=max_concurrency,
        enable_cleanup_closed=True,
    )
    return aiohttp.ClientSession(
        timeout=aiohttp_timeout,
        read_bufsize=BENCH_AIOHTTP_READ_BUFSIZE_BYTES,
        connector=connector,
        headers={"Authorization": "Bearer " + api_key},
    )
