"""Describe an object with the trained model and watch the runner pick words.

    python describe.py                      # a random held-back object
    python describe.py slot twisted glass crown
    python describe.py --facts "it is glass . it is a cradle . it is square . it is called the lantern ."
    python describe.py cup round steel needle --force round
    python describe.py --random 5           # five random held-back objects
    python describe.py --words               # list the words the model knows

The model only knows the 46 words in vocab.json, so an object is a noun, a shape,
a material, and a name from those lists (run with --words to see them).

For each object the script shows what greedy decoding writes, what beam search writes
(num_beams=4, length_penalty=0.0, the same settings as evaluate.py), the model's
probabilities for the word after "a", and its probability for each of the seven
well-formed descriptions.
"""
import argparse
import itertools
import json
import os
import random
import sys

import torch
from transformers import GPT2LMHeadModel
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()
HERE = os.path.dirname(os.path.abspath(__file__))

NOUNS = ["slot", "cradle", "tray", "hook", "ring", "rod", "cup", "lever", "plate", "clamp"]
SHAPES = ["tapered", "curved", "round", "square", "flat", "hollow", "narrow", "twisted"]
MATERIALS = ["steel", "clay", "oak", "brass", "glass", "stone", "copper", "bone"]
NAMES = ["throat", "saddle", "crown", "anchor", "needle", "bridge", "lantern", "spine"]

w2i = json.load(open(os.path.join(HERE, "vocab.json")))
i2w = {i: w for w, i in w2i.items()}
END = w2i["end"]
model = GPT2LMHeadModel.from_pretrained(os.path.join(HERE, "model")).eval()


def enc(text):
    unknown = [w for w in text.split() if w not in w2i]
    if unknown:
        sys.exit(f"The model doesn't know these words: {' '.join(unknown)}. "
                 "Run with --words to see the 46 it knows.")
    return torch.tensor([[w2i[w] for w in text.split()]])


def generate(prompt, num_beams):
    ids = enc(prompt)
    with torch.no_grad():
        out = model.generate(
            ids, attention_mask=torch.ones_like(ids), max_new_tokens=12,
            repetition_penalty=1.0, length_penalty=0.0, do_sample=False,
            num_beams=num_beams, eos_token_id=END, pad_token_id=w2i["<pad>"],
        )
    words = []
    for t in out[0, ids.shape[1]:].tolist():
        if t == END:
            break
        words.append(i2w[t])
    return " ".join(words)


def next_word_probs(prompt, k=5):
    with torch.no_grad():
        logits = model(enc(prompt)).logits[0, -1]
    probs = torch.softmax(logits, dim=-1)
    top = torch.topk(probs, k)
    return [(i2w[i.item()], p.item()) for p, i in zip(top.values, top.indices)]


def sentence_prob(prompt, sentence):
    """Product of P(word | prompt + preceding words) over the sentence's words."""
    p = enc(prompt)[0].tolist()
    s = [w2i[w] for w in sentence.split()]
    ids = torch.tensor([p + s])
    with torch.no_grad():
        logp = torch.log_softmax(model(ids).logits[0], dim=-1)
    return sum(logp[len(p) - 1 + k, t].item() for k, t in enumerate(s))


def seven_descriptions(noun, shape, material, name):
    clean = f"{shape} {material} {noun} called the {name} ."
    parts = {"shape": shape, "material": material, "name": f"called the {name}"}
    noun_first = [f"{noun} , " + " , ".join(parts[p] for p in order) + " ."
                  for order in itertools.permutations(["shape", "material", "name"])]
    return [clean] + noun_first


def fragment_facts(noun, shape, material, name, rng):
    facts = [f"{noun} ;", f"{material} ;", f"{shape} ;", f"called the {name} ;"]
    rng.shuffle(facts)
    return " ".join(facts)


def describe(prompt_facts, obj=None, force=None):
    prompt = f"facts : {prompt_facts} describe : a"
    print(f"\nprompt:  {prompt}")
    if force:
        forced = prompt + " " + force
        print(f"forced:  a {force} {generate(forced, 1)}    (you picked '{force}', greedy finished)")
    print(f"greedy:  a {generate(prompt, 1)}")
    print(f"beam-4:  a {generate(prompt, 4)}")
    print("\nthe word after 'a', by probability:")
    for w, p in next_word_probs(prompt):
        print(f"    {p:6.3f}  {w}")
    if obj:
        print("\nthe seven well-formed descriptions, by whole-sentence probability:")
        rows = [(sentence_prob(prompt, s), s) for s in seven_descriptions(*obj)]
        rows.sort(reverse=True)
        greedy_out = generate(prompt, 1)
        for lp, s in rows:
            tag = "   <- what greedy wrote" if s == greedy_out else ""
            print(f"    {torch.tensor(lp).exp().item():6.3f}  a {s}{tag}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("object", nargs="*", help="noun shape material name")
    ap.add_argument("--facts", help="the facts part of the prompt, written out yourself")
    ap.add_argument("--force", metavar="WORD", help="type the word after 'a' yourself and let greedy finish")
    ap.add_argument("--random", type=int, metavar="N", help="describe N random held-back objects")
    ap.add_argument("--words", action="store_true", help="list the words the model knows and exit")
    ap.add_argument("--seed", type=int, default=None, help="seed for shuffling the facts")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    if a.words:
        print("nouns:    ", " ".join(NOUNS))
        print("shapes:   ", " ".join(SHAPES))
        print("materials:", " ".join(MATERIALS))
        print("names:    ", " ".join(NAMES))
        print("other:    ", " ".join(w for w in w2i if w not in NOUNS + SHAPES + MATERIALS + NAMES))
        return

    if a.facts:
        words = a.facts.split()
        found = [[w for w in words if w in lst] for lst in (NOUNS, SHAPES, MATERIALS, NAMES)]
        obj = tuple(f[0] for f in found) if all(len(f) == 1 for f in found) else None
        describe(a.facts.strip(), obj=obj, force=a.force)
        return

    held = [l.split() for l in open(os.path.join(HERE, "held-back-objects.txt"))
            if l.strip() and not l.startswith("#")]

    if a.object:
        if len(a.object) != 4:
            sys.exit("Give four words: noun shape material name (or use --facts).")
        objs = [a.object]
    else:
        objs = rng.sample(held, a.random or 1)

    for noun, shape, material, name in objs:
        obj = (noun, shape, material, name)
        seen = "held back from training" if list(obj) in held else "used in training"
        print(f"\n=== {shape} {material} {noun} called the {name}  ({seen}) ===")
        describe(fragment_facts(*obj, rng), obj=obj, force=a.force)


if __name__ == "__main__":
    main()
