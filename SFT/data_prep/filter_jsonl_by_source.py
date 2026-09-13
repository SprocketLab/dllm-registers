#!/usr/bin/env python3
"""Filter a JSONL dataset by its source_type field."""

import argparse
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL path.")
    parser.add_argument("--output", required=True, help="Filtered output JSONL path.")
    parser.add_argument(
        "--source-type",
        required=True,
        help="Comma-separated source_type values to keep (or to exclude when --exclude is set).",
    )
    parser.add_argument(
        "--exclude",
        action="store_true",
        help="Invert the match: keep rows whose source_type is NOT in --source-type.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of kept rows, useful for smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")

    total = 0
    kept = 0
    missing_source = 0

    keep_types = set(args.source_type.split(","))

    with input_path.open("r", encoding="utf-8") as fin, tmp_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            row = json.loads(line)
            if "source_type" not in row:
                missing_source += 1
            in_set = row.get("source_type") in keep_types
            if args.exclude == in_set:
                continue

            fout.write(line if line.endswith("\n") else line + "\n")
            kept += 1
            if args.limit is not None and kept >= args.limit:
                break

    os.replace(tmp_path, output_path)
    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "source_type": args.source_type,
                "total_seen": total,
                "kept": kept,
                "missing_source_type": missing_source,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
