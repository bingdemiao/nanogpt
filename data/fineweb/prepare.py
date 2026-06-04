"""FineWeb-Edu (sample-10BT) → GPT-2 BPE tokens.

Same on-disk format as the openwebtext prep so train.py can switch via
``--dataset=fineweb``. The 10BT sample is the conventional nanoGPT-scale
modern web corpus.

Run once::

    python data/fineweb/prepare.py

Produces ~10B training tokens (~20GB at uint16).
"""

import os
from tqdm import tqdm

import numpy as np
import tiktoken
from datasets import load_dataset

NUM_PROC = 8
NUM_PROC_LOAD = 8

enc = tiktoken.get_encoding("gpt2")
here = os.path.dirname(__file__)


def process(example):
    ids = enc.encode_ordinary(example["text"])
    ids.append(enc.eot_token)
    return {"ids": ids, "len": len(ids)}


if __name__ == "__main__":
    # FineWeb-Edu sample-10BT split
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        num_proc=NUM_PROC_LOAD,
    )
    split_dataset = dataset.train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset["val"] = split_dataset.pop("test")

    tokenized = split_dataset.map(
        process,
        remove_columns=[c for c in dataset.column_names if c != "text"] + ["text"],
        desc="tokenizing the splits",
        num_proc=NUM_PROC,
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = os.path.join(here, f"{split}.bin")
        arr = np.memmap(filename, dtype=np.uint16, mode="w+", shape=(arr_len,))
        total_batches = 1024
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            batch = dset.shard(num_shards=total_batches, index=batch_idx,
                               contiguous=True).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            arr[idx:idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
