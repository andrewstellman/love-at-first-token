"""Check that beam search honours length_penalty (transformers warned it 'may be ignored').

Run from the repo root with the venv's python:  python check_length_penalty.py
"""
import json
import os
import random

import torch
from transformers import GPT2LMHeadModel
from transformers.utils import logging as hf_logging

HERE = os.path.dirname(os.path.abspath(__file__))
w2i = json.load(open(os.path.join(HERE, "vocab.json")))
model = GPT2LMHeadModel.from_pretrained(os.path.join(HERE, "model")).eval()
objs = [l.split() for l in open(os.path.join(HERE, "held-back-objects.txt"))
        if l.strip() and not l.startswith("#")]
rng = random.Random(0)  # same shuffle as evaluate.py
prompts = []
for noun, shape, material, name in objs:
    f = [shape, noun, f"called the {name}", material]
    rng.shuffle(f)
    prompts.append("facts : " + " ; ".join(f) + " ; describe : a")
prompts = prompts[:300]


def gen(p, nb, lp):
    ids = torch.tensor([[w2i[w] for w in p.split()]])
    with torch.no_grad():
        o = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=12,
                           repetition_penalty=1.0, length_penalty=lp, do_sample=False,
                           num_beams=nb, eos_token_id=w2i["end"], pad_token_id=0)
    return tuple(o[0].tolist())


hf_logging.set_verbosity_warning()
for nb in (1, 4):
    print(f"--- one call with num_beams={nb}, length_penalty=0.0 (any warning appears below)", flush=True)
    gen(prompts[0], nb, 0.0)
hf_logging.set_verbosity_error()

base = [gen(p, 4, 0.0) for p in prompts]
for lp in (1.0, 5.0, -5.0):
    r = [gen(p, 4, lp) for p in prompts]
    print(f"beam-4 length_penalty={lp}: outputs that differ from length_penalty=0.0: "
          f"{sum(a != b for a, b in zip(r, base))}/300", flush=True)
