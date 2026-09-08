import json

from ai_infra_bench.performance.struct import FinishReason


def test_finish_reason_preserves_string_enum_behavior():
    assert FinishReason.STOP == "stop"
    assert str(FinishReason.STOP) == "stop"
    assert f"{FinishReason.STOP}" == "stop"
    assert json.dumps(FinishReason.STOP) == '"stop"'
