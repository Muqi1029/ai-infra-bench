import pytest

from ai_infra_bench.correctness.eval_dataset.utils import resolve_dataset_path


def test_resolve_dataset_path_reports_missing_packaged_resource(monkeypatch):
    class MissingResource:
        def __truediv__(self, _resource_path):
            return self

        def exists(self):
            return False

    monkeypatch.setattr(
        "ai_infra_bench.correctness.eval_dataset.utils.resources.files",
        lambda _package: MissingResource(),
    )

    with pytest.raises(FileNotFoundError, match="newer ai-infra-bench-dataset"):
        resolve_dataset_path("data-package://data/gpqa/gpqa_diamond.csv")
