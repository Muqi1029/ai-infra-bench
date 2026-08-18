import json

import pytest

from ai_infra_bench.cli import main as cli_main
from ai_infra_bench.utils import draw
from ai_infra_bench.utils.draw import export_metric_tables_html


def _metrics(label, throughput, concurrency=1, ttft=10):
    return {
        "label": label,
        "max_concurrency": concurrency,
        "Benchmark Summary": [
            {"Metric": "Max concurrency", "Value": str(concurrency)},
            {
                "Metric": "Mean finished requests per second",
                "Value": f"{throughput / 10} req/s",
            },
            {"Metric": "Duration", "Value": f"{concurrency * 2} s"},
            {"Metric": "Output throughput", "Value": f"{throughput} tokens/s"},
            {
                "Metric": "Total cached tokens device",
                "Value": "20 (25.00%) tokens",
            },
        ],
        "Latency & Token Metrics": [
            {"Metric": "TTFT", "Mean": str(ttft), "P99": str(ttft * 2), "Unit": "ms"}
        ],
        "Finish Reason Statistics": [
            {"Finish reason": "stop", "Requests": 2, "Percentage": "100.00%"}
        ],
        "Spec Tokens Statistics": [
            {"Metric": "Total Spec Correct Drafts Histogram", "Value": [1, 2, 1]}
        ],
    }


def test_export_metric_tables_html_from_jsonl(tmp_path):
    source = tmp_path / "metrics.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(_metrics(label, throughput, concurrency, ttft))
            for label, throughput, concurrency, ttft in (
                ("run-a", 100, 1, 10),
                ("run-a", 180, 2, 18),
                ("run-b", 120, 1, 12),
                ("run-b", 210, 2, 21),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "comparison.html"

    result = export_metric_tables_html(source, output, sample_size=1, seed=7)

    assert result == output
    document = output.read_text(encoding="utf-8")
    assert "https://cdn.plot.ly/plotly-2.35.2.min.js" in document
    assert "Randomize" in document
    assert "Export selected HTML" in document
    assert "AI Infra Bench &copy; 2026" in document
    assert 'id="label-select"' in document
    assert 'id="metric-select"' not in document
    assert '"selected_labels":["run-b"]' in document

    data_start = document.index('<script id="metrics-data" type="application/json">')
    data_start = document.index(">", data_start) + 1
    data_end = document.index("</script>", data_start)
    page_data = json.loads(document[data_start:data_end])
    assert page_data["labels"] == ["run-a", "run-b"]
    categories = {category["name"]: category for category in page_data["categories"]}
    summary_metrics = {
        metric["name"]: metric for metric in categories["Benchmark Summary"]["metrics"]
    }
    assert summary_metrics["Output throughput"]["series"]["run-a"] == [
        {"x": 1.0, "y": 100.0},
        {"x": 2.0, "y": 180.0},
    ]
    assert "Total cached tokens device / Ratio" in summary_metrics
    latency_names = {
        metric["name"] for metric in categories["Latency & Token Metrics"]["metrics"]
    }
    assert latency_names == {"TTFT / Mean", "TTFT / P99"}
    histogram_names = {
        metric["name"] for metric in categories["Spec Tokens Statistics"]["metrics"]
    }
    assert histogram_names == {
        "Total Spec Correct Drafts Histogram / bin 0",
        "Total Spec Correct Drafts Histogram / bin 1",
        "Total Spec Correct Drafts Histogram / bin 2",
    }


def test_export_metric_tables_html_accepts_mapping_and_validates_sample_size(tmp_path):
    output = tmp_path / "comparison.html"

    export_metric_tables_html(_metrics("single", 100), output)

    assert output.exists()
    assert "single" in output.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="sample_size must be >= 1"):
        export_metric_tables_html([], output, sample_size=0)


def test_dashboard_defaults_to_first_label_and_groups_metrics(tmp_path):
    records = [_metrics("run-a", 100), _metrics("run-b", 120)]
    for record in records:
        record["Latency & Token Metrics"].extend(
            [
                {"Metric": "Prompt tokens", "Mean": "64", "Unit": "tokens"},
                {"Metric": "Cached tokens", "Mean": "16", "Unit": "tokens"},
            ]
        )
    output = tmp_path / "comparison.html"

    export_metric_tables_html(records, output)

    document = output.read_text(encoding="utf-8")
    assert 'id="sample-size"' not in document
    assert "color-scheme: light" in document
    assert "background: #edf3fb" in document
    assert 'className = "metric-title"' in document
    assert "groupMetrics" in document
    data_start = document.index('<script id="metrics-data" type="application/json">')
    data_start = document.index(">", data_start) + 1
    data_end = document.index("</script>", data_start)
    page_data = json.loads(document[data_start:data_end])
    assert page_data["selected_labels"] == ["run-a"]

    metrics = {
        metric["name"]: metric
        for category in page_data["categories"]
        for metric in category["metrics"]
    }
    assert metrics["TTFT / Mean"]["sections"] == ["Latency"]
    assert metrics["Prompt tokens / Mean"]["sections"] == ["Token Usage"]
    assert metrics["Cached tokens / Mean"]["sections"] == ["Cache"]
    assert metrics["Duration"]["sections"] == ["Overview"]
    assert metrics["Output throughput"]["sections"] == ["Overview"]
    assert metrics["Mean finished requests per second"]["sections"] == ["Overview"]
    assert metrics["Total cached tokens device"]["sections"] == ["Cache"]
    assert metrics["Total cached tokens device"]["unit"] == "tokens"
    assert "Max concurrency" not in metrics


def test_plot_metrics_command_uses_default_output_path(tmp_path, capsys):
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(json.dumps(_metrics("run-a", 100)) + "\n", encoding="utf-8")

    assert draw.main([str(metrics_path), "--sample-size", "1", "--seed", "7"]) == 0

    output_path = metrics_path.with_suffix(".html")
    assert output_path.exists()
    assert str(output_path) in capsys.readouterr().out


def test_cli_dispatches_plot_metrics(monkeypatch):
    calls = []
    monkeypatch.setattr(draw, "main", lambda argv: calls.append(argv) or 0)

    assert cli_main(["plot-metrics", "metrics.jsonl", "-o", "metrics.html"]) == 0
    assert calls == [["metrics.jsonl", "-o", "metrics.html"]]


def test_legacy_timestamp_labels_are_grouped_and_repeats_are_averaged(tmp_path):
    output = tmp_path / "comparison.html"
    records = [
        _metrics("server(2026-08-16 10:00:00)", 100, concurrency=2),
        _metrics("server(2026-08-16 10:00:01)", 120, concurrency=2),
        _metrics("server(2026-08-16 10:00:02)", 180, concurrency=4),
    ]

    export_metric_tables_html(records, output)

    document = output.read_text(encoding="utf-8")
    data_start = document.index('<script id="metrics-data" type="application/json">')
    data_start = document.index(">", data_start) + 1
    data_end = document.index("</script>", data_start)
    page_data = json.loads(document[data_start:data_end])
    assert page_data["labels"] == ["server"]
    summary = page_data["categories"][0]
    throughput = next(
        metric for metric in summary["metrics"] if metric["name"] == "Output throughput"
    )
    assert throughput["series"]["server"] == [
        {"x": 2.0, "y": 110.0},
        {"x": 4.0, "y": 180.0},
    ]
