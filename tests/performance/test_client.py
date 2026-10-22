import asyncio
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ai_infra_bench.utils.client import _create_bench_client_session


def test_expired_connection_is_not_reused_after_event_loop_stall():
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = 0.1

        def do_POST(self):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"OK")

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()

    async def run():
        url = f"http://127.0.0.1:{server.server_port}"
        async with _create_bench_client_session(1, request_timeout=5) as session:
            async with session.post(f"{url}/flush_cache") as response:
                assert await response.text() == "OK"

            # Payload preparation can prevent the loop from observing the
            # server's idle disconnect. Deliberately block instead of awaiting.
            time.sleep(1.2)

            async with session.post(f"{url}/v1/chat/completions") as response:
                assert response.status == 200
                assert await response.text() == "OK"

    try:
        asyncio.run(run())
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
