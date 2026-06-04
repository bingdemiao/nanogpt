"""OpenWebText → GPT-2 BPE tokens, sharded ``train.bin``/``val.bin``.

Mirrors Karpathy's nanoGPT ``data/openwebtext/prepare.py``. Requires
``datasets`` and ``tiktoken`` in the environment, plus ~54GB free.

Run once (single process is fine; ``num_proc`` parallelizes tokenization)::

    python data/openwebtext/prepare.py

Produces ~9B training tokens (~17GB at uint16).
"""

import os
from tqdm import tqdm

import numpy as np
import tiktoken
from datasets import load_dataset

NUM_PROC = 8           # tokenization workers
NUM_PROC_LOAD = 8      # download workers

enc = tiktoken.get_encoding("gpt2")
here = os.path.dirname(__file__)


def process(example):
    ids = enc.encode_ordinary(example["text"])
    ids.append(enc.eot_token)  # GPT-2 EOT (50256) as document boundary
    return {"ids": ids, "len": len(ids)}


if __name__ == "__main__":
    dataset = load_dataset("openwebtext", num_proc=NUM_PROC_LOAD, trust_remote_code=True)
    split_dataset = dataset["train"].train_test_split(
        test_size=0.0005, seed=2357, shuffle=True,
    )
    split_dataset["val"] = split_dataset.pop("test")

    tokenized = split_dataset.map(
        process,
        remove_columns=["text"],
        desc="tokenizing the splits",
        num_proc=NUM_PROC,
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = os.path.join(here, f"{split}.bin")
        dtype = np.uint16  # GPT-2 vocab fits in 16 bits
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        total_batches = 1024
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            batch = dset.shard(num_shards=total_batches, index=batch_idx,
                               contiguous=True).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            arr[idx:idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
