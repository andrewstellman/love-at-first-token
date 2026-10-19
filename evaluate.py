"""Steps 4-9: copy check, greedy vs beam-4, fork odds, whole-sentence scores, forced shape.

Run from the repo root with the venv's python, after train.py:  python evaluate.py
"""
import itertools
import json
import os
import random

import torch
from transformers import GPT2LMHeadModel

HERE = os.path.dirname(os.path.abspath(__file__))
HELD = os.path.join(HERE, "held-back-objects.txt")
REPORT = os.path.join(HERE, "REPORT.md")  # a fresh run writes here; the original run is in results/REPORT.md
SEED = 0


def report(text=""):
    print(text)
    with open(REPORT, "a") as f:
        f.write(text + "\n")


w2i = json.load(open(os.path.join(HERE, "vocab.json")))
i2w = {i: w for w, i in w2i.items()}
END = w2i["end"]
model = GPT2LMHeadModel.from_pretrained(os.path.join(HERE, "model")).eval()
stats = json.load(open(os.path.join(HERE, "train_stats.json")))

objects = []
for line in open(HELD):
    if line.startswith("#") or not line.strip():
        continue
    noun, shape, material, name = line.split()
    objects.append(dict(noun=noun, shape=shape, material=material, name=name))

rng = random.Random(SEED)
for o in objects:  # one shuffled fact order per object, reused by every step
    facts = [o["shape"], o["noun"], f"called the {o['name']}", o["material"]]
    rng.shuffle(facts)
    o["prompt"] = "facts : " + " ; ".join(facts) + " ; describe : a"
objects = objects[:300]


def enc(text):
    return torch.tensor([[w2i[w] for w in text.split()]])


def clean(o):
    return f"a {o['shape']} {o['material']} {o['noun']} called the {o['name']} ."


def generate(prompt, **kw):
    ids = enc(prompt)
    with torch.no_grad():
        out = model.generate(
            ids, attention_mask=torch.ones_like(ids), max_new_tokens=12,
            repetition_penalty=1.0, length_penalty=0.0, do_sample=False,
            eos_token_id=END, pad_token_id=w2i["<pad>"], **kw,
        )
    new = []
    for t in out[0, ids.shape[1]:].tolist():
        if t == END:
            break
        new.append(i2w[t])
    # the description starts with the prompt's final "a" (plus any forced words)
    lead = prompt.split("describe :")[1].split()
    return " ".join(lead + new)


def greedy(prompt):
    return generate(prompt, num_beams=1)


def beam4(prompt):
    return generate(prompt, num_beams=4)


def score(prompt, sentence_tokens):
    """Sum of log P(token | prompt + preceding tokens) over sentence_tokens."""
    p = enc(prompt)[0].tolist()
    s = [w2i[w] for w in sentence_tokens]
    ids = torch.tensor([p + s])
    with torch.no_grad():
        logp = torch.log_softmax(model(ids).logits[0], dim=-1)
    return sum(logp[len(p) - 1 + k, t].item() for k, t in enumerate(s))


# ---- Step 4: copy check ------------------------------------------------------
report("\n## Step 4 — copy check (prompt forced to `describe : a <shape>`, greedy finishes)\n")
report("Held-back objects are taken in file order; each object's four facts are shuffled once "
       "with `random.Random(0)` and that order is reused in every step.\n")
report("```")
copied = 0
for o in objects[:20]:
    out = greedy(o["prompt"] + " " + o["shape"])
    ok = o["material"] in out.split() and f"called the {o['name']}" in out
    copied += ok
    report(f"{'OK  ' if ok else 'MISS'} [{o['noun']} {o['shape']} {o['material']} {o['name']}]  ->  {out}")
report("```\n")
report(f"Own material and name both present: **{copied}/20**.")
if copied <= 10:
    report("\n**Stopped:** the model has not learned to copy facts from the prompt, so the "
           "decoder comparison would be meaningless. Steps 5-9 were not run.")
    raise SystemExit(1)
report("The model copies facts from the prompt; continuing.")

# ---- Steps 5-6: greedy vs beam-4 ----------------------------------------------
report("\n## Steps 5-6 — greedy vs beam-4 on 300 held-back objects\n")
report("`model.generate(max_new_tokens=12, repetition_penalty=1.0, length_penalty=0.0, "
       "do_sample=False, eos_token_id=<end>)`, with `num_beams=1` for (a) and `num_beams=4` for (b). "
       "Nothing else differs.\n")
for o in objects:
    o["greedy"] = greedy(o["prompt"])
    o["beam"] = beam4(o["prompt"])
g_clean = sum(o["greedy"] == clean(o) for o in objects) / len(objects) * 100
b_clean = sum(o["beam"] == clean(o) for o in objects) / len(objects) * 100
report(f"- (a) greedy: **{g_clean:.1f}%** clean ({sum(o['greedy'] == clean(o) for o in objects)}/300)")
report(f"- (b) beam-4: **{b_clean:.1f}%** clean ({sum(o['beam'] == clean(o) for o in objects)}/300)\n")
report("Five examples:\n")
report("| object (noun shape material name) | prompt | greedy | beam-4 |")
report("|---|---|---|---|")
for o in objects[:5]:
    report(f"| {o['noun']} {o['shape']} {o['material']} {o['name']} | `{o['prompt']}` "
           f"| `{o['greedy']}` | `{o['beam']}` |")

# ---- Step 7: odds at the fork ------------------------------------------------
report("\n## Step 7 — the fork (next token after `describe : a`)\n")
p_noun, p_shape, noun_ahead = [], [], 0
for o in objects:
    with torch.no_grad():
        probs = torch.softmax(model(enc(o["prompt"])).logits[0, -1], dim=-1)
    pn, ps = probs[w2i[o["noun"]]].item(), probs[w2i[o["shape"]]].item()
    p_noun.append(pn)
    p_shape.append(ps)
    noun_ahead += pn > ps
mean_noun, mean_shape = sum(p_noun) / 300, sum(p_shape) / 300
report(f"- mean P(correct noun) = **{mean_noun:.3f}**")
report(f"- mean P(correct shape) = **{mean_shape:.3f}**")
report(f"- noun ahead of shape in **{noun_ahead}/300** objects")

# ---- Step 8: whole-sentence scores -------------------------------------------
report("\n## Step 8 — the model grades seven whole sentences per object\n")
report("Score = sum of log-probabilities of every token after the prompt's final `a`, "
       "through the closing `.` and the `end` token, each given the prompt and the preceding tokens.\n")


def candidates(o):
    out = {"clean": clean(o).split()[1:]}
    parts = [[o["shape"]], [o["material"]], ["called", "the", o["name"]]]
    for perm in itertools.permutations(range(3)):
        toks = [o["noun"]]
        for k in perm:
            toks += [","] + parts[k]
        out["noun-first " + "/".join(("shape", "material", "name")[k] for k in perm)] = toks + ["."]
    return {k: v + ["end"] for k, v in out.items()}


clean_top, nf_objects, nf_clean_higher = 0, 0, 0
for o in objects:
    sc = {k: score(o["prompt"], v) for k, v in candidates(o).items()}
    o["scores"] = sc
    clean_top += max(sc, key=sc.get) == "clean"
    g = o["greedy"].split()
    if len(g) > 1 and g[1] == o["noun"]:  # greedy's description is noun-first
        nf_objects += 1
        nf_clean_higher += sc["clean"] > score(o["prompt"], g[1:] + ["end"])
report(f"- clean sentence scores highest of the seven: **{clean_top}/300 ({clean_top / 3:.1f}%)**")
report(f"- greedy wrote a noun-first sentence: {nf_objects}/300; of those, clean scored higher "
       f"than greedy's actual sentence: **{nf_clean_higher}/{nf_objects}**\n")
o = objects[0]
report(f"Example object: {o['noun']} {o['shape']} {o['material']} {o['name']}, prompt `{o['prompt']}`, "
       f"greedy wrote `{o['greedy']}`\n")
report("| candidate | sentence | log-prob |")
report("|---|---|---|")
for k, v in candidates(o).items():
    report(f"| {k} | `a {' '.join(v)}` | {o['scores'][k]:.3f} |")

# ---- Step 9: one forced word -------------------------------------------------
report("\n## Step 9 — one forced word (prompt ends `describe : a <shape>`, greedy finishes)\n")
forced = [greedy(o["prompt"] + " " + o["shape"]) for o in objects[:50]]
f_clean = sum(f == clean(o) for f, o in zip(forced, objects)) / 50 * 100
report(f"Clean: **{f_clean:.1f}%** ({sum(f == clean(o) for f, o in zip(forced, objects))}/50)")

# ---- Summary -----------------------------------------------------------------
size = os.path.getsize(os.path.join(HERE, "model", "model.safetensors"))
report("\n## Summary\n")
report("| measure | value |")
report("|---|---|")
report(f"| greedy % clean (300) | {g_clean:.1f}% |")
report(f"| beam-4 % clean (300) | {b_clean:.1f}% |")
report(f"| forced-shape % clean (50) | {f_clean:.1f}% |")
report(f"| mean P(noun) at the fork | {mean_noun:.3f} |")
report(f"| mean P(shape) at the fork | {mean_shape:.3f} |")
report(f"| % objects where clean sentence scored highest | {clean_top / 3:.1f}% |")
report(f"| greedy wrote noun-first but clean scored higher | {nf_clean_higher} of {nf_objects} |")
report(f"| total training time | {stats['train_seconds']:.1f} s |")
report(f"| final step count | {stats['final_step']} |")
report(f"| model.safetensors size | {size:,} bytes |")
