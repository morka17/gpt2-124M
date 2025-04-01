import os 
import json 
import requests 
import tiktoken 
from tqdm import tqdm 
import torch 
import torch.nn as nn 
from torch.nn import functional as F 
from transformers import GPT2LMHeadModeL



#  ==================================================
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

def download_file(url: str, fname: str, chunk_size=1024):
    """
    Helper function to download a file from a url and save it to a local file.
    """
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname, 
        total=total, 
        unit="iB", 
        unit_scale=True, 
        unit_divisor=1024, 
        ) as bar:
            for data in resp.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                bar.update(size)

hellaswags = {
     "train": "https://raw.githubusercontent.com/hellaswag/hellaswag/master/data/hellaswag_train.jsonl",
     "val": "https://raw.githubusercontent.com/hellaswag/hellaswag/master/data/hellaswag_val.jsonl",
     "test": "https://raw.githubusercontent.com/hellaswag/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")

def download(split: str):
     """ Download the hellaswag dataset and save it to the cache directory."""
     os.makedirs(DATA_CACHE_DIR, exist_ok=True)
     data_url = hellaswags[split]
     data_filename = os.path.join(DATA_CACHE_DIR, f"helloswag_{split}.jsonl")
     if not os.path.exists(data_filename):
          print(f"Downloading {data_filename} to {data_filename}...")
          download_file(data_url, data_filename)
     else:
          print(f"{data_filename} already exists.")


def render_example(example: dict):
    """ Render a single example from the hellaswag dataset."""
    ctx =  example["ctx"]
    label = example["label"]
    endings = example["endings"]

    # data needed to reproduce this eval on the C size 
    data = {
         "label": label,
         "ctx_token": ctx,
         "endings_tokens": [],
    }

    # gather up all the tokens 
    ctx_tokens = enc.encode(ctx)
    data["ctx_token"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings: 
        end_tokens = enc.encode(" " + end ) # note: prepending " " to GPT2 TOKENIZER 
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(end_tokens) * [1]*len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((1, max_len), dtype=torch.long)
    mask = torch.zeros((1, max_len), dtype=torch.long)
    for i, (row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(row)] = torch.tensor(row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label
    

def iterate_examples(split: str):
    # 10, 042 train examples 
    # 1, 000 val examples 
    download(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        for line in f:
            example = json.loads(line)
            yield example 

def get_most_likely_row(tokens, mask, logits):
    """
    Calculate which option is most likely based on the model's logits and masking.
    
    Args:
        tokens (torch.Tensor): Input tokens tensor
        mask (torch.Tensor): Mask tensor indicating which positions to evaluate
        logits (torch.Tensor): Model output logits
        
    Returns:
        int: Index of the most likely option
    """
    # Evaluate the autoregressive model loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_labels = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_labels = shift_labels.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_labels, reduction="none")

    # Apply mask to losses
    shift_mask = (mask[..., 1:]).contiguous()
    masked_shift_losses = shift_losses * shift_mask
    
    # Calculate average loss per option
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    
    # Return the index of the option with lowest average loss
    return avg_loss.argmin().item()

@torch.no_grad()
def evaluate(mode_type, device):
    
    torch.set_float32_matmul_precision("high")  # use tf32 on matmul 
    model = GPT2LMHeadModeL.from_pretrained(mode_type)
    model.to(device)
    # model = torch.compile(mode)

    num_correct = 0
    num_total = 0   
    num_correct_norm = 0
    for example in iterate_examples("val"):
        data, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)


        # get the logits 
        logits = model(tokens).logits 
        # evaluate the autoregressive model loss at all positions 
        pred = get_most_likely_row(tokens, mask, logits)

        # accumulate stats 
        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred == label)
        
        print(f"{num_total} accum_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")
        
    # acc = num_correct / num_total
    # acc_norm = num_correct_norm / num_total
    # return acc, acc_norm











