import json
import os
import re
from typing import List,Dict
from tqdm import tqdm

from .utils_io import load_yaml, save_jsonl,split_dataset
from .logger import get_logger

logger = get_logger("preprocess_pubmedqa_full")

_WS = re.compile(r"\s+")

def _clean_text(t: str) -> str:
    return _WS.sub(" ", (t or "").strip())

def preprocess_pubmedqa_full(cfg_path: str = "data_pipeline/config_full.yaml"):
    """
    Reads data/raw/pubmedqa_full/train.jsonl, cleans fields, applies
    length filters, and writes JSONL files in data/processed/pubmedqa_full/:
      - sft_full_train.jsonl
      - sft_full_valid.jsonl
      - sft_full_test.jsonl
      - sft_full.jsonl (combined)
    Output schema: {instruction, input, output, meta}
    """
    cfg=load_yaml(cfg_path)
    raw_path=os.path.join(cfg["raw_dir"],"train.jsonl")
    processed_dir=cfg["processed_dir"]
    os.makedirs(processed_dir,exist_ok=True)

    min_w = int(cfg.get("min_context_words", 10))
    max_w = int(cfg.get("max_context_words", 500))

    logger.info(f"Loading raw file: {raw_path}")
    records: List[Dict] = []

    with open(raw_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Processing full samples"):
            row = json.loads(line)

            question = _clean_text(row.get("question", ""))

            # Handle both formats:
            # 1) {"context": {"contexts": ["...", "..."]}}
            # 2) {"context": "long abstract ..."}
            context_field = row.get("context", "")
            if isinstance(context_field, dict):
                contexts = context_field.get("contexts", []) or []
                ctx = _clean_text(" ".join(_clean_text(c) for c in contexts))
            else:
                ctx = _clean_text(context_field)

            # drop too-short contexts
            words = ctx.split()
            if len(words) < min_w:
                continue
            if len(words) > max_w:
                ctx = " ".join(words[:max_w])

            sample = {
                "instruction": question,
                "input": ctx,
                "output": "",  # unlabeled → model will generate later (self-labeling or SFT w/ empty target)
                "meta": {
                    "pubid": row.get("pubid", "") or row.get("id", ""),
                    "split": "train"
                },
            }
            records.append(sample)

    n = len(records)
    logger.info(f"Cleaned {n} samples (~{n/1e6:.2f}M)")

    train, valid, test = split_dataset(
        records,
        (cfg["train_split"], cfg["valid_split"], cfg["test_split"]),
        seed=cfg["seed"],
    )

    from os.path import join
    save_jsonl(train, join(processed_dir, "sft_full_train.jsonl"))
    save_jsonl(valid, join(processed_dir, "sft_full_valid.jsonl"))
    save_jsonl(test,  join(processed_dir, "sft_full_test.jsonl"))
    save_jsonl(records, cfg["sft_output_file"])

    logger.info(f"Saved processed files to: {processed_dir}")
    logger.info(f"train={len(train)}  valid={len(valid)}  test={len(test)}")
    return {
        "train": len(train),
        "valid": len(valid),
        "test": len(test),
        "total": len(records),
    }

if __name__ == "__main__":
    preprocess_pubmedqa_full()


