"""
Autoresearch pretraining script — MLX native for Apple Silicon.
Port of karpathy/autoresearch train.py from PyTorch to MLX.
Usage: uv run train.py
"""

import gc
import math
import time
from dataclasses import dataclass, asdict

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


_rms_norm_weights = {}

def rms_norm(x):
    d = x.shape[-1]
    if d not in _rms_norm_weights:
        _rms_norm_weights[d] = mx.ones((d,))
    return mx.fast.rms_norm(x, weight=_rms_norm_weights[d], eps=1e-5)


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    # x: (B, T, H, D)
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return mx.concatenate([y1, y2], axis=3)


_sliding_window_mask_cache = {}

def _get_sliding_window_mask(T, window_size):
    """Get or create a cached sliding window causal mask (only for window < T)."""
    key = (T, window_size)
    if key not in _sliding_window_mask_cache:
        rows = mx.arange(T).reshape(T, 1)
        cols = mx.arange(T).reshape(1, T)
        valid = mx.logical_and(cols <= rows, (rows - cols) < window_size)
        mask = mx.where(valid, mx.zeros((T, T)), mx.full((T, T), float('-inf')))
        _sliding_window_mask_cache[key] = mask
    return _sliding_window_mask_cache[key]


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self._has_ve = has_ve(layer_idx, config.n_layer)
        if self._has_ve:
            self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)

    def __call__(self, x, ve, cos, sin, window_size):
        B, T, C = x.shape
        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.reshape(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * mx.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            # gate: (B, T, n_kv_head) -> (B, T, n_kv_head, 1)
            v = v + mx.expand_dims(gate, axis=-1) * ve

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = rms_norm(q)
        k = rms_norm(k)

        # Transpose to (B, H, T, D) for scaled_dot_product_attention
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scale = self.head_dim ** -0.5

        # Use native "causal" string for full attention (hardware-accelerated, no T×T mask)
        # Only build explicit mask for sliding window (window < T)
        if window_size > 0 and window_size < T:
            mask = _get_sliding_window_mask(T, window_size)
            y = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        else:
            y = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")

        # Transpose back to (B, T, H, D) and reshape
        y = y.transpose(0, 2, 1, 3).reshape(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def __call__(self, x):
        x = self.c_fc(x)
        x = mx.square(nn.relu(x))  # ReluSquared
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def __call__(self, x, ve, cos, sin, window_size):
        x = x + self.attn(rms_norm(x), ve, cos, sin, window_size)
        x = x + self.mlp(rms_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = [Block(config, i) for i in range(config.n_layer)]
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.resid_lambdas = mx.ones((config.n_layer,))
        self.x0_lambdas = mx.full((config.n_layer,), 0.1)

        # Value embeddings — use list with None for layers without VE
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self._ve_indices = []
        ve_list = []
        for i in range(config.n_layer):
            if has_ve(i, config.n_layer):
                self._ve_indices.append(i)
                ve_list.append(nn.Embedding(config.vocab_size, kv_dim))
        self.value_embeds = ve_list
        self._ve_layer_map = {idx: pos for pos, idx in enumerate(self._ve_indices)}

        # Rotary embeddings (precomputed)
        self.rotary_seq_len = config.sequence_len * 10
        self._cos, self._sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000):
        channel_range = mx.arange(0, head_dim, 2).astype(mx.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = mx.arange(seq_len).astype(mx.float32)
        freqs = mx.outer(t, inv_freq)
        cos = mx.cos(freqs).astype(mx.bfloat16)
        sin = mx.sin(freqs).astype(mx.bfloat16)
        # (1, T, 1, D/2)
        cos = cos.reshape(1, seq_len, 1, -1)
        sin = sin.reshape(1, seq_len, 1, -1)
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": long_window, "S": short_window}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Last layer always full attention
        window_sizes[-1] = long_window
        return window_sizes

    def init_weights(self):
        """Initialize all weights to match the original PyTorch implementation."""
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5

        # Embedding: normal(0, 1) in bfloat16
        self.wte.weight = mx.random.normal(shape=self.wte.weight.shape).astype(mx.bfloat16)

        # lm_head: normal(0, 0.001)
        self.lm_head.weight = mx.random.normal(shape=self.lm_head.weight.shape) * 0.001

        # Transformer blocks
        for block in self.blocks:
            # Attention q,k,v: uniform(-s, s)
            for layer in [block.attn.c_q, block.attn.c_k, block.attn.c_v]:
                layer.weight = mx.random.uniform(low=-s, high=s, shape=layer.weight.shape)
            # Attention c_proj: zeros
            block.attn.c_proj.weight = mx.zeros_like(block.attn.c_proj.weight)
            # MLP c_fc: uniform(-s, s)
            block.mlp.c_fc.weight = mx.random.uniform(low=-s, high=s, shape=block.mlp.c_fc.weight.shape)
            # MLP c_proj: zeros
            block.mlp.c_proj.weight = mx.zeros_like(block.mlp.c_proj.weight)
            # VE gate: zeros
            if block.attn._has_ve:
                block.attn.ve_gate.weight = mx.zeros_like(block.attn.ve_gate.weight)

        # Per-layer scalars
        self.resid_lambdas = mx.ones((self.config.n_layer,))
        self.x0_lambdas = mx.full((self.config.n_layer,), 0.1)

        # Value embeddings: uniform(-s, s), bfloat16
        for ve in self.value_embeds:
            ve.weight = mx.random.uniform(low=-s, high=s, shape=ve.weight.shape).astype(mx.bfloat16)

        # Recompute rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        self._cos, self._sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = self.num_params()
        value_embeds_numel = sum(ve.weight.size for ve in self.value_embeds)
        nparams_exclude = (self.wte.weight.size + value_embeds_numel +
                          self.resid_lambdas.size + self.x0_lambdas.size)
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            effective_seq = min(window_size, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_params(self):
        params = self.parameters()
        leaves = mlx.utils.tree_flatten(params)
        return sum(v.size for _, v in leaves)

    def __call__(self, idx, targets=None, reduction='mean'):
        B, T = idx.shape
        cos = self._cos[:, :T]
        sin = self._sin[:, :T]

        x = self.wte(idx)
        x = rms_norm(x)
        x0 = x

        for i, block in enumerate(self.blocks):
            rl = self.resid_lambdas[i]
            xl = self.x0_lambdas[i]
            x = rl * x + xl * x0
            ve = self.value_embeds[self._ve_layer_map[i]](idx) if i in self._ve_layer_map else None
            x = block(x, ve, cos, sin, self.window_sizes[i])

        x = rms_norm(x)

        softcap = 30.0
        logits = self.lm_head(x)
        logits = logits.astype(mx.float32)
        logits = softcap * mx.tanh(logits / softcap)

        if targets is not None:
            # Cross entropy loss
            logits_flat = logits.reshape(-1, logits.shape[-1])
            targets_flat = targets.reshape(-1)
            # ignore_index=-1: mask out padding
            mask = targets_flat != -1
            loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction='none')
            if reduction == 'mean':
                loss = mx.sum(loss * mask) / mx.maximum(mx.sum(mask), 1)
            elif reduction == 'none':
                loss = (loss * mask).reshape(targets.shape)
            return loss
        return logits


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

ASPECT_RATIO = 64
HEAD_DIM = 128
WINDOW_PATTERN = "SSSL"

TOTAL_BATCH_SIZE = 2**13    # ~8K tokens per step (tuned for 8GB Mac)
EMBEDDING_LR = 0.6
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.04
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.2
ADAM_BETAS = (0.9, 0.99)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.3
FINAL_LR_FRAC = 0.0

DEPTH = 2                   # optimized for 8GB Mac
DEVICE_BATCH_SIZE = 4       # small batch for memory

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
mx.random.seed(42)

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")


def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )


config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")

model = GPT(config)
model.init_weights()
mx.eval(model.parameters())

num_params = model.num_params()
num_flops_per_token = model.estimate_flops()
print(f"Num parameters: {num_params:,}")
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

# ---------------------------------------------------------------------------
# Build optimizer param groups
# ---------------------------------------------------------------------------

model_dim = config.n_embd
dmodel_lr_scale = (model_dim / 768) ** -0.5
print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")

# ---------------------------------------------------------------------------
# Build MultiOptimizer: Muon for 2D block weights, AdamW for everything else
# ---------------------------------------------------------------------------

ADAMW_INITIAL_LR = 0.008
MUON_INITIAL_LR = 0.04

def is_muon_param(path, param):
    """Muon for 2D weight matrices in transformer blocks only."""
    return "blocks" in path and "weight" in path and param.ndim == 2

muon_opt = optim.Muon(
    learning_rate=MUON_INITIAL_LR,
    momentum=0.95,
    weight_decay=0.0,  # handle decay separately via schedule
    nesterov=True,
    ns_steps=5,
)

adamw_opt = optim.AdamW(
    learning_rate=ADAMW_INITIAL_LR,
    betas=list(ADAM_BETAS),
    eps=1e-10,
    weight_decay=0.01,
)

optimizer = optim.MultiOptimizer(
    optimizers=[muon_opt, adamw_opt],
    filters=[is_muon_param],
)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# ---------------------------------------------------------------------------
# LR Schedules
# ---------------------------------------------------------------------------

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC


def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def loss_fn(model, x, y):
    return model(x, y)


loss_and_grad_fn = nn.value_and_grad(model, loss_fn)


# Compile the step function: forward + backward + optimizer update
# Use state-based compilation to handle mutable model and optimizer
state = [model.state, optimizer.state]

def _step(x, y):
    loss, grads = loss_and_grad_fn(model, x, y)
    optimizer.update(model, grads)
    return loss

compiled_step = mx.compile(_step, inputs=state, outputs=state)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

while True:
    t0 = time.time()

    # Progress and schedules (update LR before step)
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)

    muon_opt.learning_rate = MUON_INITIAL_LR * lrm
    adamw_opt.learning_rate = ADAMW_INITIAL_LR * lrm

    # Compiled training step: forward + backward + optimizer update in one fused dispatch
    if grad_accum_steps == 1:
        avg_loss = compiled_step(x, y)
    else:
        # Fallback for gradient accumulation
        accumulated_grads = None
        total_loss = mx.array(0.0)
        for micro_step in range(grad_accum_steps):
            loss, grads = loss_and_grad_fn(model, x, y)
            total_loss = total_loss + loss
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = mlx.utils.tree_map(
                    lambda a, g: a + g, accumulated_grads, grads
                )
            if micro_step < grad_accum_steps - 1:
                x, y, epoch = next(train_loader)
        accumulated_grads = mlx.utils.tree_map(
            lambda g: g * (1.0 / grad_accum_steps), accumulated_grads
        )
        optimizer.update(model, accumulated_grads)
        avg_loss = total_loss / grad_accum_steps

    mx.eval(model.parameters(), optimizer.state, avg_loss)

    # Prefetch next batch
    x, y, epoch = next(train_loader)

    train_loss_f = avg_loss.item()

    # Fast fail
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    # GC management
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE

# ---------------------------------------------------------------------------
# Final eval
# ---------------------------------------------------------------------------

EVAL_BATCH_SIZE = 8  # eval batch size
val_bpb = evaluate_bpb(model, tokenizer, EVAL_BATCH_SIZE)

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

t_end = time.time()

try:
    peak_memory_mb = mx.metal.get_peak_memory() / 1024 / 1024
except AttributeError:
    peak_memory_mb = 0.0

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_memory_mb:   {peak_memory_mb:.1f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
