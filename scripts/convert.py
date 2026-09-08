"""Convert timestamped payload dumps into a JSONL file.

Each matched JSON file is a list of ``[timestamp, payload_json]`` pairs.
Records are merged, sorted by timestamp, and written as one payload per line.
"""

import logging
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

from ai_infra_bench.performance.bench import read_requests_with_ts
from ai_infra_bench.utils.io import _dump_jsonl

logger = logging.getLogger(__name__)


def parse_args(args=None):
    parser = ArgumentParser(
        description=(
            "Convert timestamped payload dump JSON files into a single JSONL "
            "file of request payloads."
        ),
        epilog=(
            "examples:\n"
            "  python scripts/convert.py \\\n"
            "    --payload-regex-path 'dumps/**/*.json' \\\n"
            "    --output-path out/requests.jsonl\n"
        ),
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--payload-regex-path",
        required=True,
        metavar="GLOB",
        help=(
            "Glob matching timestamped dump JSON files. Each file is a JSON "
            "array of [timestamp, payload_json] pairs "
            "(timestamp format: YYYY-MM-DD_HH-MM-SS.ffffff)."
        ),
    )
    parser.add_argument(
        "--output-path",
        required=True,
        metavar="FILE",
        help="Output JSONL path. Parent directories are created if missing.",
    )
    return parser.parse_args(args)


def main():
    args = parse_args()
    requests = read_requests_with_ts(args.payload_regex_path)
    if not requests:
        logger.error(
            "Read 0 requests. Check --payload-regex-path (%s)",
            args.payload_regex_path,
        )
        sys.exit(1)

    path = Path(args.output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    _dump_jsonl(requests, path)
    logger.info("Successfully dumped %d requests to %s", len(requests), path)


if __name__ == "__main__":
    main()
