# Reader experiment v2 — Claude Code run

Run date: 2026-10-17 11:10 EDT. Machine: Darwin arm64, CPU only.

## Setup

venv at `v2/.venv`, installed with `pip install torch transformers safetensors`.

```
python 3.14.6
torch 2.14.1
transformers 5.20.1
safetensors 0.8.0
```

## Step 1 — tokenize on spaces

Lines: 20000. Distinct words: 46. Vocabulary size including one `<pad>` token for batching: 47. Longest line: 35 tokens.

Word-to-ID table (also saved as `v2/vocab.json`):

```
<pad>=0  ,=1  .=2  :=3  ;=4  a=5  anchor=6  bone=7  brass=8  bridge=9  called=10  clamp=11  clay=12  copper=13  cradle=14  crown=15  cup=16  curved=17  describe=18  end=19  facts=20  flat=21  glass=22  hollow=23  hook=24  is=25  it=26  lantern=27  lever=28  narrow=29  needle=30  oak=31  plate=32  ring=33  rod=34  round=35  saddle=36  slot=37  spine=38  square=39  steel=40  stone=41  tapered=42  the=43  throat=44  tray=45  twisted=46
```

## Step 2 — train

GPT2Config: n_layer=2, n_head=2, n_embd=32, n_positions=40, vocab_size=47, resid/embd/attn_pdrop=0.0, tie_word_embeddings=False. Parameters: 29,760. Adam lr 3e-3, batch 128 (random lines each step), seed 0.

Loss printed every 500 steps is the mean training loss over those 500 steps. Stopping rule: train 6000 steps; then, while the mean loss of the last 500 steps is more than 1% below the mean loss of the 500 steps ending 2,000 steps earlier, train another 2000 steps, up to 20000 total.

```
step    500  loss 0.9175  (7s)
step   1000  loss 0.6031  (14s)
step   1500  loss 0.5687  (21s)
step   2000  loss 0.5121  (28s)
step   2500  loss 0.5067  (35s)
step   3000  loss 0.5051  (42s)
step   3500  loss 0.5041  (49s)
step   4000  loss 0.5037  (56s)
step   4500  loss 0.5030  (62s)
step   5000  loss 0.5025  (69s)
step   5500  loss 0.5024  (77s)
step   6000  loss 0.5018  (85s)
-- at step 6000: loss 0.5018 vs 0.5037 at step 4000 (+0.38% improvement) -> flat
```

Final step count: **6000**. Training time: **85.1 s**.

## Step 3 — save_pretrained

Files in `v2/model/`:

```
       816 bytes  config.json
       218 bytes  generation_config.json
   121,712 bytes  model.safetensors
```

`model.safetensors` header (2664 bytes of JSON after the 8-byte length):

```
__metadata__  {'format': 'pt'}
lm_head.weight                           [47, 32]     F32
transformer.h.0.attn.c_attn.bias         [96]         F32
transformer.h.0.attn.c_attn.weight       [32, 96]     F32
transformer.h.0.attn.c_proj.bias         [32]         F32
transformer.h.0.attn.c_proj.weight       [32, 32]     F32
transformer.h.0.ln_1.bias                [32]         F32
transformer.h.0.ln_1.weight              [32]         F32
transformer.h.0.ln_2.bias                [32]         F32
transformer.h.0.ln_2.weight              [32]         F32
transformer.h.0.mlp.c_fc.bias            [128]        F32
transformer.h.0.mlp.c_fc.weight          [32, 128]    F32
transformer.h.0.mlp.c_proj.bias          [32]         F32
transformer.h.0.mlp.c_proj.weight        [128, 32]    F32
transformer.h.1.attn.c_attn.bias         [96]         F32
transformer.h.1.attn.c_attn.weight       [32, 96]     F32
transformer.h.1.attn.c_proj.bias         [32]         F32
transformer.h.1.attn.c_proj.weight       [32, 32]     F32
transformer.h.1.ln_1.bias                [32]         F32
transformer.h.1.ln_1.weight              [32]         F32
transformer.h.1.ln_2.bias                [32]         F32
transformer.h.1.ln_2.weight              [32]         F32
transformer.h.1.mlp.c_fc.bias            [128]        F32
transformer.h.1.mlp.c_fc.weight          [32, 128]    F32
transformer.h.1.mlp.c_proj.bias          [32]         F32
transformer.h.1.mlp.c_proj.weight        [128, 32]    F32
transformer.ln_f.bias                    [32]         F32
transformer.ln_f.weight                  [32]         F32
transformer.wpe.weight                   [40, 32]     F32
transformer.wte.weight                   [47, 32]     F32
```

`config.json`:

```json
{
  "activation_function": "gelu_new",
  "add_cross_attention": false,
  "architectures": [
    "GPT2LMHeadModel"
  ],
  "attn_pdrop": 0.0,
  "bos_token_id": 19,
  "dtype": "float32",
  "embd_pdrop": 0.0,
  "eos_token_id": 19,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "model_type": "gpt2",
  "n_embd": 32,
  "n_head": 2,
  "n_inner": null,
  "n_layer": 2,
  "n_positions": 40,
  "pad_token_id": 0,
  "reorder_and_upcast_attn": false,
  "resid_pdrop": 0.0,
  "scale_attn_by_inverse_layer_idx": false,
  "scale_attn_weights": true,
  "summary_activation": null,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": true,
  "summary_type": "cls_index",
  "summary_use_proj": true,
  "tie_word_embeddings": false,
  "transformers_version": "5.20.1",
  "use_cache": true,
  "vocab_size": 47
}
```

## Step 4 — copy check (prompt forced to `describe : a <shape>`, greedy finishes)

Held-back objects are taken in file order; each object's four facts are shuffled once with `random.Random(0)` and that order is reused in every step.

```
OK   [slot twisted glass crown]  ->  a twisted glass slot called the crown .
OK   [rod narrow clay lantern]  ->  a narrow clay rod called the lantern .
OK   [clamp round oak needle]  ->  a round oak clamp called the needle .
OK   [tray hollow clay saddle]  ->  a hollow clay tray called the saddle .
OK   [rod tapered copper crown]  ->  a tapered copper rod called the crown .
OK   [ring round brass crown]  ->  a round brass ring called the crown .
OK   [slot tapered copper crown]  ->  a tapered copper slot called the crown .
OK   [lever flat brass bridge]  ->  a flat brass lever called the bridge .
OK   [lever twisted brass lantern]  ->  a twisted brass lever called the lantern .
OK   [plate square brass crown]  ->  a square brass plate called the crown .
OK   [cradle flat stone spine]  ->  a flat stone cradle called the spine .
OK   [cup hollow oak throat]  ->  a hollow oak cup called the throat .
OK   [hook flat bone throat]  ->  a flat bone hook called the throat .
OK   [lever tapered glass lantern]  ->  a tapered glass lever called the lantern .
OK   [tray square brass bridge]  ->  a square brass tray called the bridge .
OK   [slot hollow glass needle]  ->  a hollow glass slot called the needle .
OK   [tray twisted oak lantern]  ->  a twisted oak tray called the lantern .
OK   [plate flat glass throat]  ->  a flat glass plate called the throat .
OK   [ring square stone anchor]  ->  a square stone ring called the anchor .
OK   [slot tapered copper lantern]  ->  a tapered copper slot called the lantern .
```

Own material and name both present: **20/20**.
The model copies facts from the prompt; continuing.

## Steps 5-6 — greedy vs beam-4 on 300 held-back objects

`model.generate(max_new_tokens=12, repetition_penalty=1.0, length_penalty=0.0, do_sample=False, eos_token_id=<end>)`, with `num_beams=1` for (a) and `num_beams=4` for (b). Nothing else differs.

- (a) greedy: **19.3%** clean (58/300)
- (b) beam-4: **98.0%** clean (294/300)

Five examples:

| object (noun shape material name) | prompt | greedy | beam-4 |
|---|---|---|---|
| slot twisted glass crown | `facts : called the crown ; twisted ; slot ; glass ; describe : a` | `a slot , called the crown , glass , twisted .` | `a twisted glass slot called the crown .` |
| rod narrow clay lantern | `facts : narrow ; rod ; clay ; called the lantern ; describe : a` | `a rod , clay , narrow , called the lantern .` | `a narrow clay rod called the lantern .` |
| clamp round oak needle | `facts : round ; called the needle ; clamp ; oak ; describe : a` | `a clamp , oak , round , called the needle .` | `a round oak clamp called the needle .` |
| tray hollow clay saddle | `facts : tray ; hollow ; clay ; called the saddle ; describe : a` | `a hollow clay tray called the saddle .` | `a hollow clay tray called the saddle .` |
| rod tapered copper crown | `facts : called the crown ; tapered ; copper ; rod ; describe : a` | `a rod , copper , called the crown , tapered .` | `a tapered copper rod called the crown .` |

## Step 7 — the fork (next token after `describe : a`)

- mean P(correct noun) = **0.597**
- mean P(correct shape) = **0.397**
- noun ahead of shape in **242/300** objects

## Step 8 — the model grades seven whole sentences per object

Score = sum of log-probabilities of every token after the prompt's final `a`, through the closing `.` and the `end` token, each given the prompt and the preceding tokens.

- clean sentence scores highest of the seven: **294/300 (98.0%)**
- greedy wrote a noun-first sentence: 242/300; of those, clean scored higher than greedy's actual sentence: **236/242**

Example object: slot twisted glass crown, prompt `facts : called the crown ; twisted ; slot ; glass ; describe : a`, greedy wrote `a slot , called the crown , glass , twisted .`

| candidate | sentence | log-prob |
|---|---|---|
| clean | `a twisted glass slot called the crown . end` | -1.626 |
| noun-first shape/material/name | `a slot , twisted , glass , called the crown . end` | -2.045 |
| noun-first shape/name/material | `a slot , twisted , called the crown , glass . end` | -2.244 |
| noun-first material/shape/name | `a slot , glass , twisted , called the crown . end` | -1.921 |
| noun-first material/name/shape | `a slot , glass , called the crown , twisted . end` | -2.188 |
| noun-first name/shape/material | `a slot , called the crown , twisted , glass . end` | -2.009 |
| noun-first name/material/shape | `a slot , called the crown , glass , twisted . end` | -1.835 |

## Step 9 — one forced word (prompt ends `describe : a <shape>`, greedy finishes)

Clean: **100.0%** (50/50)

## Summary

| measure | value |
|---|---|
| greedy % clean (300) | 19.3% |
| beam-4 % clean (300) | 98.0% |
| forced-shape % clean (50) | 100.0% |
| mean P(noun) at the fork | 0.597 |
| mean P(shape) at the fork | 0.397 |
| % objects where clean sentence scored highest | 98.0% |
| greedy wrote noun-first but clean scored higher | 236 of 242 |
| total training time | 85.1 s |
| final step count | 6000 |
| model.safetensors size | 121,712 bytes |

## Note — the `length_penalty` warning

During steps 5–9, transformers 5.20.1 printed this once: `The following generation flags are not valid and may be ignored: ['length_penalty']`. Because a warning could mean beam search had silently ignored `length_penalty=0.0`, `v2/check_length_penalty.py` checked it after the run. Output:

```
--- one call with num_beams=1, length_penalty=0.0 (any warning appears below)
[transformers] The following generation flags are not valid and may be ignored: ['length_penalty']. ...
--- one call with num_beams=4, length_penalty=0.0 (any warning appears below)
beam-4 length_penalty=1.0: outputs that differ from length_penalty=0.0: 40/300
beam-4 length_penalty=5.0: outputs that differ from length_penalty=0.0: 285/300
beam-4 length_penalty=-5.0: outputs that differ from length_penalty=0.0: 8/300
```

The warning comes only from the greedy call, where `length_penalty` doesn't apply. Beam search did use the setting: changing it changes the outputs. So the beam-4 numbers above were produced with `length_penalty=0.0` as specified. (At the default 1.0, 40 of the 300 beam outputs would have come out differently.)

## Choices not specified in the prompt

- I added a `<pad>` token (ID 0) so lines of different lengths fit in one batch. It is masked out of the loss and attention, which brings the vocabulary to 47 (46 words + pad). `end` (ID 19) is both the eos and the bos ID.
- What "flat" means: the mean loss of the last 500 steps improved less than 1% compared with the 500 steps ending 2,000 steps earlier. At step 6,000 the improvement was 0.38%, so training stopped at 6,000.
- Each batch draws 128 random lines. The seed is 0 everywhere.
- Held-back objects are used in file order: the first 20 for step 4, the first 300 for steps 5–8 and the first 50 for step 9. Steps 4 and 9 therefore overlap on 20 objects. Each object's fact order is shuffled once with `random.Random(0)` and reused in every step.
- Step 4 passes if both the material and `called the <name>` appear in at least 11 of the 20 outputs. The result was 20/20.
- Step 8 scores the tokens after the prompt's `a`, through `.` and `end`. "Greedy wrote noun-first" means greedy's first word after `a` was the noun.
