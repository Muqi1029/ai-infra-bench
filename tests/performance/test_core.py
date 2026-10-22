import asyncio

from ai_infra_bench.performance.core import request_func


class FakeContent:
    async def iter_any(self):
        yield (
            b'data: {"choices":[{"text":"hello","finish_reason":null}]}\n\n'
            b'data: {"choices":[{"text":"","finish_reason":"length"}],'
            b'"usage":{"prompt_tokens":4,"completion_tokens":1}}\n\n'
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
    assert output.content == "hello"
    assert output.ttft_ms > 0
    assert output.finish_reason == "length"
    assert output.prompt_tokens == 4
    assert output.completion_tokens == 1
    assert payload == {"prompt": [1, 2, 3, 4], "return_meta_info": True}
    assert "return_meta_info" not in output.payload


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
