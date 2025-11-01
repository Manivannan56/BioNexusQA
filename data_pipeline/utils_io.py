import os
import json
import yaml
import random
from pathlib import Path
from typing import Iterable, List, Tuple, Any, Dict


def load_yaml(path:str)-> Dict[str,Any]:
    with open(path,"r") as f:
        return yaml.safe_load(f)
    
def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def save_jsonl(records:Iterable[Dict[str,Any]],filepath:str)->None:
    """
    Writes an iterable of dicts to JSONL. Ensures parent dir exists.
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath,"w",encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r,ensure_ascii="False")+"\n")

def split_dataset(records: List[Dict[str, Any]],
                  ratios: Tuple[float, float, float],
                  seed: int = 42):
    """
    Randomly splits records into train/valid/test by ratios.
    ratios must sum to <= 1.0 (we use remaining for test).
    """
    assert len(ratios) == 3, "ratios must be (train, valid, test)"
    random.seed(seed)
    recs = list(records)  # copy
    random.shuffle(recs)
    n = len(recs)
    n_train = int(n * ratios[0])
    n_valid = int(n * ratios[1])
    train = recs[:n_train]
    valid = recs[n_train:n_train + n_valid]
    test = recs[n_train + n_valid:]
    return train, valid, test

