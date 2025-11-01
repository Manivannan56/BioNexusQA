import os
import json
import logging
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

def main():
    logging.info("Downloading PubMedQA (pqa_artificial) from Hugging Face ...")

    dataset = load_dataset("pubmed_qa", "pqa_artificial")
    os.makedirs("data/raw/pubmedqa_artificial", exist_ok=True)

    out_file = "data/raw/pubmedqa_artificial/train.jsonl"
    with open(out_file, "w") as f:
        for item in dataset["train"]:
            json.dump({
                "question": item.get("question", ""),
                "context": item.get("context", ""),
                "id": item.get("id", "")
            }, f)
            f.write("\n")

    logging.info(f"Saved {out_file} ({len(dataset['train'])} rows)")
    logging.info("PubMedQA artificial download complete.")

if __name__ == "__main__":
    main()
