import os, json, logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

def merge_jsonl_files(files, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    count = 0
    with open(output_file, "w") as out:
        for path in files:
            with open(path) as f:
                for line in f:
                    out.write(line)
                    count += 1
    logging.info(f"Merged {len(files)} files into {output_file} ({count} total rows)")

def main():
    input_files = [
        "data/raw/pubmedqa_unlabeled/train.jsonl",
        "data/raw/pubmedqa_artificial/train.jsonl",
    ]
    output_file = "data/raw/pubmedqa_full/train.jsonl"
    merge_jsonl_files(input_files, output_file)

if __name__ == "__main__":
    main()
