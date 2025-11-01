"""
dataset_split.py
----------------
Splits `sft_full_train.json` into:
    - dapt_200k.json   (for DAPT)
    - sft_remaining.json  (for later SFT)
Usage:
    python dataset_split.py --input data/sft_full_train.json --output_dir data/ --dapt_size 200000
"""

import json
import random
import os
import argparse
from tqdm import tqdm

def load_jsonl_or_json(path):
    """Loads a list of JSON objects (supports both .json and .jsonl)."""
    print(f"[INFO] Loading dataset from {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            data = [json.loads(line) for line in f]
        else:
            data = json.load(f)
    print(f"[INFO] Loaded {len(data):,} samples.")
    return data


def save_json(data, path):
    """Saves data as JSON (pretty formatted)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved {len(data):,} samples to {path}")


def main(args):
    random.seed(args.seed)

    # Load data
    data = load_jsonl_or_json(args.input)
    total = len(data)

    if args.dapt_size >= total:
        raise ValueError(f"DAPT size {args.dapt_size} exceeds total dataset size {total}")

    # Shuffle for randomness
    print("[INFO] Shuffling data ...")
    random.shuffle(data)

    # Split
    dapt_data = data[:args.dapt_size]
    remaining_data = data[args.dapt_size:]

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Save splits
    save_json(dapt_data, os.path.join(args.output_dir, "dapt_200k.jsonl"))
    save_json(remaining_data, os.path.join(args.output_dir, "sft_full_train.jsonl"))

    print(f"[DONE] Split complete:")
    print(f"        DAPT: {len(dapt_data):,}")
    print(f"        Remaining (SFT): {len(remaining_data):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset for DAPT and SFT")
    parser.add_argument("--input", type=str, required=True, help="Path to sft_full_train.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for splits")
    parser.add_argument("--dapt_size", type=int, default=200000, help="Number of samples for DAPT")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    main(args)
