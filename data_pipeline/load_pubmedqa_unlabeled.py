import os
from datasets import load_dataset
from .utils_io import ensure_dir
from .logger import get_logger


logger =get_logger("load_pubmedqa_unlabeled")
import os
from datasets import load_dataset
from .utils_io import ensure_dir
from .logger import get_logger

logger = get_logger("load_pubmedqa_unlabeled")

def download_pubmedqa_unlabeled(raw_dir: str = "data/raw/pubmedqa_unlabeled"):
    """
    Downloads the ~1M unlabeled PubMedQA subset from Hugging Face.
    Saves a single 'train.jsonl' file into data/raw/pubmedqa_unlabeled/.
    """
    ensure_dir(raw_dir)
    logger.info("Downloading PubMedQA (pqa_unlabeled) from Hugging Face ...")
    ds = load_dataset("pubmed_qa", "pqa_unlabeled",split="train",download_mode="force_redownload")  # only 'train' split exists

    out_path = os.path.join(raw_dir, "train.jsonl")
    ds.to_json(out_path, orient="records", lines=True)
    logger.info(f"Saved {out_path} ({len(ds)} rows).")
    logger.info(" PubMedQA unlabeled download complete.")
    return out_path

if __name__ == "__main__":
    download_pubmedqa_unlabeled()

