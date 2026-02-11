"""Download and convert datasets for RL baseline experiments.

Converts math and code splits from nvidia/Nemotron-Post-Training-Dataset-v1
to JSONL format expected by NeMo RL ResponseDataset.

Usage:
    uv run python experiments/rl_baselines/shared/prepare_data.py \
        --output-dir /path/to/data \
        --split math
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def convert_messages_to_input(messages: list[dict[str, str]]) -> str:
    """Extract the user prompt from a messages list.

    The Nemotron dataset stores prompts in OpenAI messages format.
    We extract the last user message as the input prompt.
    """
    for msg in reversed(messages):
        if msg["role"] == "user":
            return msg["content"]
    raise ValueError(f"No user message found in: {messages}")


def prepare_math(output_dir: Path, val_size: int = 1000) -> None:
    """Download and convert the math split."""
    print("Loading nvidia/Nemotron-Post-Training-Dataset-v1 (math split)...")
    ds = load_dataset(
        "nvidia/Nemotron-Post-Training-Dataset-v1",
        "SFT-Math",
        split="train",
    )
    print(f"Loaded {len(ds)} rows")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Split: last val_size rows for validation
    train_ds = ds.select(range(len(ds) - val_size))
    val_ds = ds.select(range(len(ds) - val_size, len(ds)))

    for split_name, split_ds in [("train", train_ds), ("val", val_ds)]:
        out_path = output_dir / f"math_{split_name}.jsonl"
        with open(out_path, "w") as f:
            for row in split_ds:
                entry = {
                    "input": convert_messages_to_input(row["messages"]),
                    "output": row.get("expected_answer", ""),
                }
                f.write(json.dumps(entry) + "\n")
        print(f"Wrote {len(split_ds)} rows to {out_path}")


def prepare_code(output_dir: Path, val_size: int = 1000) -> None:
    """Download and convert the code split."""
    print("Loading nvidia/Nemotron-Post-Training-Dataset-v1 (code split)...")
    ds = load_dataset(
        "nvidia/Nemotron-Post-Training-Dataset-v1",
        "SFT-Code",
        split="train",
    )
    print(f"Loaded {len(ds)} rows")

    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ds.select(range(len(ds) - val_size))
    val_ds = ds.select(range(len(ds) - val_size, len(ds)))

    for split_name, split_ds in [("train", train_ds), ("val", val_ds)]:
        out_path = output_dir / f"code_{split_name}.jsonl"
        with open(out_path, "w") as f:
            for row in split_ds:
                entry = {
                    "input": convert_messages_to_input(row["messages"]),
                    "output": row.get("expected_answer", ""),
                }
                f.write(json.dumps(entry) + "\n")
        print(f"Wrote {len(split_ds)} rows to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare RL baseline datasets")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--split",
        choices=["math", "code", "all"],
        default="all",
        help="Which split to prepare",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=1000,
        help="Number of rows for validation set",
    )
    args = parser.parse_args()

    if args.split in ("math", "all"):
        prepare_math(args.output_dir, args.val_size)
    if args.split in ("code", "all"):
        prepare_code(args.output_dir, args.val_size)

    print("Done.")


if __name__ == "__main__":
    main()
