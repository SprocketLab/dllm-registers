"""Download the public training JSONL; no token is needed for the public release."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/mix60k.jsonl"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new path")
    from huggingface_hub import hf_hub_download
    cached = hf_hub_download("albertge/mix60k-math-code-sft", "mix60k.jsonl",
                             revision="2fb4f8ed669d2e2fea3676b5369163aeb979e853",
                             repo_type="dataset", token=False)
    with open(cached) as handle:
        count = 0
        for line in handle:
            row = json.loads(line)
            if "question" not in row or "gpt54_reasoning_trace" not in row:
                raise ValueError("Unexpected training data schema")
            count += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(cached, "rb") as source, args.output.open("xb") as output:
        shutil.copyfileobj(source, output)
    print(f"Saved {count} examples to {args.output}")


if __name__ == "__main__":
    main()
