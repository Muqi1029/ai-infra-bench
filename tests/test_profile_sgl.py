import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from ai_infra_bench.cli import main as cli_main
from ai_infra_bench.profile_sgl import build_payload, main, send_profile


def test_build_payload_defaults():
    from argparse import Namespace

    args = Namespace(
        output_dir="/tmp/profiles",
        start_step=5,
        num_steps=5,
        activities=["CPU", "GPU"],
        profile_by_stage=False,
        with_stack=False,
        record_shapes=False,
        merge_profiles=False,
        profile_prefix=None,
        profile_stages=None,
    )
    payload = build_payload(args)
    assert payload == {
        "output_dir": "/tmp/profiles",
        "start_step": 5,
        "num_steps": 5,
        "activities": ["CPU", "GPU"],
    }


def test_build_payload_optional_flags():
    from argparse import Namespace

    args = Namespace(
        output_dir="/tmp/test",
        start_step=2,
        num_steps=10,
        activities=["CPU", "GPU", "MEM"],
        profile_by_stage=True,
        with_stack=True,
        record_shapes=True,
        merge_profiles=True,
        profile_prefix="trace",
        profile_stages=["prefill", "decode"],
    )
    payload = build_payload(args)
    assert payload["profile_by_stage"] is True
    assert payload["with_stack"] is True
    assert payload["record_shapes"] is True
    assert payload["merge_profiles"] is True
    assert payload["profile_prefix"] == "trace"
    assert payload["profile_stages"] == ["prefill", "decode"]


@pytest.mark.asyncio
async def test_send_profile_start_success():
    from argparse import Namespace

    args = Namespace(
        base_url="127.0.0.1:30000",
        api_key="EMPTY",
        stop=False,
        verbose=False,
        output_dir="/tmp/profiles",
        start_step=5,
        num_steps=5,
        activities=["CPU", "GPU"],
        profile_by_stage=False,
        with_stack=False,
        record_shapes=False,
        merge_profiles=False,
        profile_prefix=None,
        profile_stages=None,
    )

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.text = AsyncMock(return_value="ok")

    from unittest.mock import MagicMock

    mock_session = MagicMock()
    mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "ai_infra_bench.profile_sgl._create_bench_client_session",
        return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_session)),
    ):
        result = await send_profile(args)
        assert result == 0
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert call_args[0][0] == "http://127.0.0.1:30000/start_profile"
        assert call_args[1]["json"]["output_dir"] == "/tmp/profiles"


@pytest.mark.asyncio
async def test_send_profile_stop():
    from argparse import Namespace

    args = Namespace(
        base_url="127.0.0.1:30000",
        api_key="EMPTY",
        stop=True,
        verbose=False,
    )

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.text = AsyncMock(return_value="stopped")

    from unittest.mock import MagicMock

    mock_session = MagicMock()
    mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "ai_infra_bench.profile_sgl._create_bench_client_session",
        return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_session)),
    ):
        result = await send_profile(args)
        assert result == 0
        call_args = mock_session.post.call_args
        assert call_args[0][0] == "http://127.0.0.1:30000/stop_profile"
        assert call_args[1]["json"] is None


def test_cli_dispatches_profile_sgl(monkeypatch):
    calls = []
    from ai_infra_bench import profile_sgl

    monkeypatch.setattr(profile_sgl, "main", lambda argv: calls.append(argv) or 0)

    assert cli_main(["profile-sgl", "--stop"]) == 0
    assert calls == [["--stop"]]
