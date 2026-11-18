import asyncio

import pytest

from ai_infra_bench.performance.core import flush_cache, request_func


class FakeContent:
    async def iter_any(self):
        yield (
            b'data: {"choices":[{"text":"hello","finish_reason":null}]}\n\n'
            b'data: {"choices":[{"text":" world","finish_reason":"length"}],'
            b'"usage":{"prompt_tokens":4,"completion_tokens":2}}\n\n'
            b"data: [DONE]\n\n"
        )


class FakeResponse:
    status = 200
    content = FakeContent()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class FakeSession:
    def post(self, **kwargs):
        return FakeResponse()


def test_request_func_handles_completions_stream_text():
    payload = {"prompt": [1, 2, 3, 4], "return_meta_info": True}
    output = asyncio.run(
        request_func(
            FakeSession(),
            "http://localhost/v1/completions",
            payload,
            raw=True,
        )
    )

    assert output.success is True
    assert output.content == "hello world"
    assert output.ttft_ms > 0
    assert len(output.itl_ms) == 1
    assert output.itl_ms[0] >= 0
    assert output.finish_reason == "length"
    assert output.prompt_tokens == 4
    assert output.completion_tokens == 2
    assert payload == {"prompt": [1, 2, 3, 4], "return_meta_info": True}
    assert "return_meta_info" not in output.payload


def test_request_func_handles_mid_stream_error():
    class ErrorContent:
        async def iter_any(self):
            yield b'data: {"error": {"message": "engine failure"}}\n\n'

    class ErrorResponse:
        status = 200
        content = ErrorContent()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    class ErrorSession:
        def post(self, **_kwargs):
            return ErrorResponse()

    output = asyncio.run(
        request_func(ErrorSession(), "http://localhost/v1/chat/completions", {})
    )
    assert output.success is False
    assert output.error_message == "engine failure"


def test_flush_cache_skips_early_on_404():
    calls = 0

    class NotFoundResponse:
        status = 404

        async def text(self):
            return "Not Found"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    class NotFoundSession:
        def post(self, _url):
            nonlocal calls
            calls += 1
            return NotFoundResponse()

    asyncio.run(flush_cache(NotFoundSession(), "http://localhost/flush_cache"))
    assert calls == 1


def test_request_func_does_not_copy_payload_until_semaphore_is_acquired():
    class Payload(dict):
        def __deepcopy__(self, memo):
            copies.append(True)
            return dict(self)

    copies = []

    async def run():
        semaphore = asyncio.Semaphore(0)
        task = asyncio.create_task(
            request_func(
                FakeSession(), "unused", Payload(prompt="hello"), sem=semaphore
            )
        )
        await asyncio.sleep(0)
        try:
            assert copies == []
        finally:
            semaphore.release()
            output = await asyncio.wait_for(task, 2)
        assert output.success
        assert copies == [True]

    asyncio.run(run())


@pytest.mark.parametrize(
    ("status_code", "chunks", "expected_error"),
    [
        (503, [b"service busy"], "Status Code=503"),
        (200, [b""], "Empty streaming response"),
        (
            200,
            [b"<html>502 Bad Gateway</html>"],
            "Unexpected non-SSE HTTP 200 response",
        ),
        (200, [b'{"object":"error","message":"bad backend"}'], "bad backend"),
        (
            200,
            [b'event: error\ndata: {"message":"upstream timeout"}\n\n'],
            "upstream timeout",
        ),
        (
            200,
            [b'data: {"object":"error","message":"context length exceeded"}\n\n'],
            "context length exceeded",
        ),
        (
            200,
            [b'data: {"usage":{"prompt_tokens":4}}\n\ndata: [DONE]\n\n'],
            "no choices",
        ),
        (
            200,
            [
                b'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
            ],
            "prematurely",
        ),
        (
            200,
            [
                b'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"abort"}]}\n\n',
                b"data: [DONE]\n\n",
            ],
            "abort",
        ),
        (
            200,
            [
                b'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":{"type":"abort"}}]}\n\n',
                b"data: [DONE]\n\n",
            ],
            "abort",
        ),
    ],
)
def test_request_func_marks_incomplete_or_error_streams_failed(
    status_code, chunks, expected_error
):
    class MockContent:
        async def iter_any(self):
            for chunk in chunks:
                yield chunk

    class MockResponse:
        status = status_code
        content = MockContent()

        async def text(self):
            return b"".join(chunks).decode("utf-8", errors="replace")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    class MockSession:
        def post(self, **_kwargs):
            return MockResponse()

    output = asyncio.run(
        request_func(MockSession(), "http://localhost/v1/chat/completions", {})
    )
    assert output.success is False
    assert output.error_message is not None
    assert expected_error in output.error_message
