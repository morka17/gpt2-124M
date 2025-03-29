import inspect
import os
import time
import tiktoken 
from dataclasses import dataclass 
import torch 
import torch.nn as nn
from torch.nn import functional as F 
import math 
from torch.distributed import init_process_group
# ----------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd  % config.n_head == 0 
        # key, query, value projections for all heads, but in a batch 
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection 
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 
        # Regularization 
        self.n_head = config.n_head 
        self.n_embd = config.n_embd
        # not bias more mask 
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    
    def forward(self,x ):
        B, T, C = x.size()  # batch size, sequence length, embedding dimesonality (n_embd)
        # calculate Query, Key, Values, for all heads in batch and move head forward to be the batch
        # e.g in GPT-2 (124M), n_head=12, hs=64, so nh+hs=C=768 channels in the transformer 
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # attention (materialize the large (T, T) matrix for all the queries and keys)    
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, Tn, hs)

        # Flash Attention 
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side 
        # output projection 
        y = self.c_proj(y)
        return y 

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x 

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x 


@dataclass 
class GPTConfig: 
    block_size: int = 1024 
    vocab_size: int = 50257 
    n_layer: int = 6 
    n_head: int = 6 
    n_embd: int = 384 

class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config 

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), 
            wpe = nn.Embedding(config.block_size, config.n_embd), 
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), 
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.ln_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme -> parameters usability 
        self.transformer.wte.weight = self.ln_head.weight 

        # Init params
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        std = 0.02  # Move std definition here
        if isinstance(module, nn.Linear):
            if hasattr(module, 'NANOGPT_SCALE_INIT'): 
                std *= (2 * self.config.n_layer) ** -0.5  
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of the length"
        # forward the token and position embeddings 
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # Changed from (B, T) to (0, T)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of the shape (T, N)
        tok_emb = self.transformer.wte(idx)  # token embeddings of the shape (B, T, N)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer 
        for block in self.transformer.h: 
            x = block(x)
        # forward the final layernorm and the classifier 
        x = self.transformer.ln_f(x)
        logits = self.ln_head(x) # (B,T, vocab_size) 
        loss = None 
        if targets is not None: 
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss 



    @classmethod 
    def from_pretrained(cls, model_type):
        """"Loads pretrained GPT-2 model weights from hugginface """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', "gpt2-xl"}
        from transformers import GPT2LMHeadModel
        print("Loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_emb are determmined from model_type 
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768), # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600) # 155M params
        }[model_type]

        config_args['vocab_size']= 50257  # always 50257 for GPT model checkpoints 
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model 
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask / buffer 

        # init a huggingface/transformers model 
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] 
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf: 
            if any(k.endswith(w) for w in transposed):
                # special treatment for the conv1D weights we needs to transpose 
                assert sd_hf[k].shape[::-1] == sd[k].shape 
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            
            else: 
                # vanilla copy over the other parameters 
                assert sd_hf[k].shape == sd[k].shape 
                with torch.no_grad():
                    sd[k].copy_(sd_hf[K])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters which requires grad 
        param_dict = {pn:p for pn, p in self.named_parameters()}
        param_dict = {pn:p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e all weight tensors in matmuls + embeddings decay, all biases and layernorms don't
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2] 
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay}, 
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create the AdamW optimizer and use the fused version if it is available 
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters 
        use_fused = fused_available and  "cuda" in device 
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
        return optimizer 




class DataLoaderLite: 
    def __init__(self, B, T):
        self.B = B 
        self.T = T 

        # at init tokens from disk and store them in memory 
        with open("input.txt", "r") as f: 
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"I epoch = {len(self.tokens) // (B * T)} tokens")

        # state 
        self.current_position = 0 

    def next_batch(self):
        B, T = self.B, self.T 
        buf = self.tokens[self.current_position:self.current_position + B*T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        # advance the position in the tensor 
        self.current_position += B * T 
        # if loading the next batch would be out of bounds, reset 
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0 
        return x, y 




# set up DDP (Distributed Data Parallel)
# torchnn command sets the env variable RANK, WORLD_SIZE, LOCAL_RANK
ddp = int(os.environ.get("RANK", -1)) != -1 # check if DDP is enabled 
if ddp: 
    # use of DDP atm demands CUDA, set the device accordingly 
    assert torch.cuda.is_available(), "for now, DDP requires CUDA"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ.get("RANK"))
    ddp_world_size = int(os.environ.get("WORLD_SIZE"))
    ddp_local_rank = int(os.environ.get("LOCAL_RANK"))
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do the logging  and checkpoint saving 
else:   
    # vanilla, non-DDP run 
    ddp_rank = 0 
    ddp_local_rank = 0 
    ddp_world_size = 1 
    master_process = True 
    # attempt to autodetect a GPU if one is available 
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"using device: {device}")


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"Using device: {device}")
# device = "cpu" # OVERRIDE 


# Gradient accumulation steps 
total_batch_size = 524288  # 2^19 
B = 16 
T = 1024 
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size" 
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process: 
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> caculated gradient accumulation steps: {grad_accum_steps}")
    print(f"effective batch size: {B * T * grad_accum_steps * ddp_world_size}")

    
print(f"gpu rank: {ddp}")
print("Bye")
import sys; sys.exit(0)

train_loader = DataLoaderLite(B=B, T=T)
# train_loader = DataLoaderLite(B=4, T=32)
torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig())
model.eval()
model.to(device)
mode = torch.compile(model)

max_lr = 3e-4 
min_lr = max_lr * 0.1 
warmup_steps = 10 
max_steps = 50
def get_lr(it):
    # Linear warmup for warmup steps 
    if it < warmup_steps: 
        return max_lr + (it + 1) / warmup_steps 
    if it > max_steps: 
        return min_lr 
    
    decay_ratio =  (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1 
    coeff = 0.5 * (1.0 *  math.cos(math.pi * decay_ratio))  # coeff starts at 1 and gets to 0 
    return min_lr + coeff * (max_lr - min_lr )

# optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.2, learning_rate=6e-4, device=device)

for i in range(50):
    t0 = time.time()
    loss_accum = 0.0 
    optimizer.zero_grad()  
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
            # import code; code.interact(local=locals())
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Prevent the model from gradient shock in case of bad data  batch 
    # determine and set the learning rate for this iteration 
    lr = get_lr(i)
    for param_group in optimizer.params_groups: 
        param_group['lr'] = lr 
    optimizer.step()
    # torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) # time difference in milliseconds 
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt 
    print(f"step {i} | loss: {loss_accum.item():.4f} | lr: {lr:.6e} | norm: {norm:.4f} | dt: {dt:.2f}ms  | tokens/sec: {tokens_per_sec:.2f}")         



# print(loss)
import sys; sys.exit(0)








torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length: 
    # forward the odel to get the logits
    with torch.no_grad():
        logits = model(x)  # (B, T, vocab_size)
        logits = logits[:, -1, :]  # (B, vocab_size)
        # get the probabilities 
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)  # (B, 1)
        xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
        # append to the sequence 
        x = torch.cat([x , xcol], dim=1)



# torchrun --standalone --nproc_per_node=1 train_gpt2.py