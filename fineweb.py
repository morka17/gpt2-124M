"""
Fineweb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/fineweb-edu
Download and tokenizes the dataset and save it in the data folder
Run simply as 
$ python fineweb.py
will save shards to the local directory "edu_fineweb10B"
"""

import os 
import multiprocessing as mp 
import json 
import pandas as pd 
import tiktoken 
from datasets import load_dataset 
from tqdm import tqdm 
import numpy as np 


# ==================================
local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8) # 100M tokens per shard , total of 100 shards 


# create the cache the local directory if it doesn't exist 
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset 
fw = load_dataset("HuggingFaceFW/fineweb-edu", split="train", remote_dir=remote_name)

# init the tokenizer 
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens_encoder['<|endoftext|>'] # end of text token 
def tokenize(doc):
    # tokenize a single document and returns a numpy array of tokens 
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token values must fit in uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16 




def write_datafile(filename, tokens_np):
    # writes a numpy array of uint16 tokens to a binary file 
    with open(filename, "wb") as f: 
        f.write(tokens_np.tobytes())

# tokenize all documents and write output shards, each of shard_size tokens  
nprocs = max(1, os.cpu_count() // 2)
with mp.Pool(nprocs) as pool:
    shard_index = 0 
    # preallocate buffer to hold current shard 
    all_tokens_np = np.empty((shard_size, ), dtype=np.uint16)
    token_count = 0 
    progress_bar = None 
    for tokens in pool.imap(tokenize, fw, chunksize=16):
        # simply append tokens to current shard 
        if token_count + len(tokens) < shard_size:
            all_tokens_np[token_count:token_count+len(tokens)] = tokens 
            token_count += len(tokens)
            # update progress bar 
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"shard {shard_index:02d}")
            progress_bar.update(len(tokens))
        else:
            # write current shard 
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            # split the document into whatever fits in this shard; the remainder goes 
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1 
            progress_bar = None 
            # populate the next shard with the leftovers of the current doc 
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder 

# write any remaining tokens as the last shard 
if token_count != 0:
    split = "val" if shard_index == 0 else "train"
    filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
    write_datafile(filename, all_tokens_np[:token_count])

            














