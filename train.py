"""Steps 1-3: tokenize on spaces, train a tiny GPT-2, save it.

Run from the repo root with the venv's python:  python train.py
"""
import json
import os
import random
import struct
import time

import torch
from transformers import GPT2Config, GPT2LMHeadModel

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "training-text.txt")
MODEL_DIR = os.path.join(HERE, "model")
REPORT = os.path.join(HERE, "REPORT.md")  # a fresh run writes here; the original run is in results/REPORT.md
SEED = 0
PAD = "<pad>"


def report(text):
    print(text)
    with open(REPORT, "a") as f:
        f.write(text + "\n")


random.seed(SEED)
torch.manual_seed(SEED)

# ---- Step 1: tokenize by splitting on spaces -------------------------------
lines = [l.split() for l in open(DATA).read().splitlines() if l.strip()]
words = sorted({w for l in lines for w in l})
vocab = [PAD] + words  # <pad> fills short lines in a batch; never a training target
w2i = {w: i for i, w in enumerate(vocab)}
with open(os.path.join(HERE, "vocab.json"), "w") as f:
    json.dump(w2i, f, indent=1)
max_len = max(len(l) for l in lines)

report("\n## Step 1 — tokenize on spaces\n")
report(f"Lines: {len(lines)}. Distinct words: {len(words)}. Vocabulary size including "
       f"one `<pad>` token for batching: {len(vocab)}. Longest line: {max_len} tokens.\n")
report("Word-to-ID table (also saved as `vocab.json`):\n")
report("```\n" + "  ".join(f"{w}={i}" for w, i in w2i.items()) + "\n```")

ids = torch.full((len(lines), max_len), w2i[PAD], dtype=torch.long)
mask = torch.zeros((len(lines), max_len), dtype=torch.long)
for r, l in enumerate(lines):
    ids[r, : len(l)] = torch.tensor([w2i[w] for w in l])
    mask[r, : len(l)] = 1
labels = ids.clone()
labels[mask == 0] = -100  # padding is not predicted

# ---- Step 2: train ----------------------------------------------------------
config = GPT2Config(
    vocab_size=len(vocab), n_layer=2, n_head=2, n_embd=32, n_positions=40,
    resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, tie_word_embeddings=False,
    bos_token_id=w2i["end"], eos_token_id=w2i["end"], pad_token_id=w2i[PAD],
)
model = GPT2LMHeadModel(config)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
BATCH, FIRST, BLOCK, CAP, LOG = 128, 6000, 2000, 20000, 500
FLAT_TOL = 0.01  # "flat" = last-500-step mean loss improved < 1% over the past 2,000 steps

report("\n## Step 2 — train\n")
report(f"GPT2Config: n_layer=2, n_head=2, n_embd=32, n_positions=40, vocab_size={len(vocab)}, "
       "resid/embd/attn_pdrop=0.0, tie_word_embeddings=False. "
       f"Parameters: {sum(p.numel() for p in model.parameters()):,}. "
       "Adam lr 3e-3, batch 128 (random lines each step), seed 0.\n")
report("Loss printed every 500 steps is the mean training loss over those 500 steps. "
       f"Stopping rule: train {FIRST} steps; then, while the mean loss of the last 500 steps is more "
       f"than {FLAT_TOL:.0%} below the mean loss of the 500 steps ending 2,000 steps earlier, "
       f"train another {BLOCK} steps, up to {CAP} total.\n")
report("```")

model.train()
window_means = {}  # step -> mean loss over the 500 steps ending at that step
acc, step, target = 0.0, 0, FIRST
t0 = time.time()
while True:
    while step < target:
        b = torch.randint(0, len(lines), (BATCH,))
        out = model(input_ids=ids[b], attention_mask=mask[b], labels=labels[b])
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        step += 1
        acc += out.loss.item()
        if step % LOG == 0:
            window_means[step] = acc / LOG
            report(f"step {step:6d}  loss {window_means[step]:.4f}  ({time.time() - t0:.0f}s)")
            acc = 0.0
    now, before = window_means[step], window_means[step - BLOCK]
    still_falling = now < before * (1 - FLAT_TOL)
    report(f"-- at step {step}: loss {now:.4f} vs {before:.4f} at step {step - BLOCK} "
           f"({(before - now) / before:+.2%} improvement) -> "
           + ("still falling" if still_falling else "flat"))
    if not still_falling or step >= CAP:
        break
    target = min(step + BLOCK, CAP)
train_secs = time.time() - t0
report("```\n")
report(f"Final step count: **{step}**. Training time: **{train_secs:.1f} s**."
       + (" Stopped at the 20,000-step cap while still falling." if still_falling else ""))

# ---- Step 3: save -----------------------------------------------------------
model.save_pretrained(MODEL_DIR)
with open(os.path.join(HERE, "train_stats.json"), "w") as f:
    json.dump({"final_step": step, "train_seconds": train_secs}, f)

report("\n## Step 3 — save_pretrained\n")
report("Files in `model/`:\n\n```")
for name in sorted(os.listdir(MODEL_DIR)):
    report(f"{os.path.getsize(os.path.join(MODEL_DIR, name)):>10,} bytes  {name}")
report("```\n")

st = os.path.join(MODEL_DIR, "model.safetensors")
with open(st, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
report(f"`model.safetensors` header ({n} bytes of JSON after the 8-byte length):\n\n```")
for k, v in header.items():
    if k == "__metadata__":
        report(f"__metadata__  {v}")
    else:
        report(f"{k:40s} {str(v['shape']):12s} {v['dtype']}")
report("```\n")
report("`config.json`:\n\n```json\n" + open(os.path.join(MODEL_DIR, "config.json")).read().strip() + "\n```")
