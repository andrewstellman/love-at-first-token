# Love at First Token

A tiny experiment showing that an AI's "Yoda-speak" (details tacked onto the end of a sentence after the noun) can come from the program that runs the model, not from the model itself. The same 29,760-number model writes the awkward sentence under greedy decoding and the clean sentence under beam search, with no retraining.

This repository is the companion to the O'Reilly Radar article *Love at First Token* (link coming when it's published), which is a follow-up to [Why Does AI Write Like Yoda?](https://www.oreilly.com/radar/why-does-ai-write-like-yoda/). Everything the article reports is in here: the training text, the scripts, the trained model file, and the report from the run the numbers came from.

## The idea in one paragraph

A language model is a file of numbers. It doesn't write anything; it scores how likely each word is to come next. A separate program (the runner, or in Hugging Face's `transformers` library, the `generate` method) turns those scores into text, one word at a time. The simplest way to do that, greedy decoding, takes the highest-scoring word at every step and never looks back. This experiment builds training text where the most likely *first* word leads to a worse *whole* sentence, trains a tiny GPT-2 on it, and shows that greedy decoding falls into the trap while beam search, which keeps several candidate sentences going before committing, does not. The model scored the clean sentence highest 98% of the time either way. Only the runner changed.

## What's in the repo

| File | What it is |
|---|---|
| `training-text.txt` | 20,000 lines of training text, one example per line (see "The training text" below). |
| `held-back-objects.txt` | 1,536 objects never used in training. The scripts test on the first 300. |
| `train.py` | Tokenizes on spaces, trains the model, saves it to `model/`, prints the safetensors header. |
| `evaluate.py` | Runs greedy and beam-4 decoding on 300 held-back objects, reports the fork probabilities, scores all seven descriptions of each object, and runs the forced-first-word check. |
| `describe.py` | Try the model yourself: pick an object, see what greedy and beam search write, the probabilities for the word after *a*, and the scores for all seven descriptions. See "Trying the model yourself" below. |
| `check_length_penalty.py` | Confirms that beam search honours `length_penalty` (transformers prints a warning that it "may be ignored"; it isn't). |
| `model/` | The trained model: `model.safetensors` (121,712 bytes), `config.json`, `generation_config.json`. |
| `vocab.json` | The 47-token vocabulary (46 words and punctuation marks plus `<pad>`). |
| `train_stats.json` | Step count and training time from the original run. |
| `results/REPORT.md` | The full report from the run the article's numbers come from, written step by step by the scripts. |

## Results from the original run

All on the first 300 held-back objects, with the model in `model/`.

| | |
|---|---|
| Greedy decoding wrote the clean sentence | 58 of 300 (19%) |
| Model scored the clean sentence highest of the seven | 294 of 300 (98%) |
| Beam search (`num_beams=4`) wrote the clean sentence | 294 of 300 (98%) |
| Forced first word (shape typed in, greedy finishes) | 50 of 50 |
| Average probability of the noun as the word after *a* | 0.597 |
| Average probability of the shape as the word after *a* | 0.397 |
| Training | 6,000 steps, 85 seconds on a laptop CPU |

The worked example in the article is the held-back object *twisted glass slot called the crown*. The model's probabilities for its seven possible descriptions:

| description | probability |
|---|---|
| a twisted glass slot called the crown . | 0.197 |
| a slot , called the crown , glass , twisted . (what greedy wrote) | 0.160 |
| a slot , glass , twisted , called the crown . | 0.146 |
| a slot , called the crown , twisted , glass . | 0.134 |
| a slot , twisted , glass , called the crown . | 0.129 |
| a slot , glass , called the crown , twisted . | 0.112 |
| a slot , twisted , called the crown , glass . | 0.106 |

## Running it yourself

You need Python 3.10 or later and a CPU. No GPU required.

```sh
python3 -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt    # torch, transformers, safetensors
python train.py                    # about two minutes on a laptop; writes model/ and REPORT.md
python evaluate.py                 # about a minute; appends the results to REPORT.md
python check_length_penalty.py     # optional
```

`train.py` overwrites `model/` and starts a new `REPORT.md` in the repo root. The original run's report is kept in `results/REPORT.md`, and the original model is the one committed in `model/`, so if you want to keep it, copy it somewhere before retraining. To evaluate the committed model without retraining, run `evaluate.py` on its own.

Your numbers will differ a little from the table above. Training starts from random weights, so a retrained model learns slightly different probabilities. The pattern holds: greedy low, beam search near the model's own ranking, forced first word clean every time. Across the runs I've done, greedy has landed anywhere from 2% to 19% clean and beam search from 85% to 98%.

The original run used Python 3.14.6, torch 2.14.1, transformers 5.20.1, and safetensors 0.8.0.

## Trying the model yourself

`describe.py` loads the committed model and lets you hand it objects. It needs the same venv as the scripts above (torch and transformers), but it doesn't need you to retrain anything.

```sh
python describe.py                                 # a random held-back object
python describe.py slot twisted glass crown        # noun shape material name
python describe.py cup round steel needle --force round   # you pick the word after "a", greedy finishes
python describe.py --random 5                      # five random held-back objects
python describe.py --words                         # the 46 words the model knows
```

For each object it prints the prompt, what greedy decoding writes, what beam search writes, the model's probabilities for the word after *a*, and its probability for each of the seven well-formed descriptions, with the one greedy wrote marked. For example:

```
=== twisted glass slot called the crown  (held back from training) ===

prompt:  facts : twisted ; slot ; glass ; called the crown ; describe : a
greedy:  a slot , glass , twisted , called the crown .
beam-4:  a twisted glass slot called the crown .

the word after 'a', by probability:
     0.812  slot
     0.184  twisted
     0.001  rod

the seven well-formed descriptions, by whole-sentence probability:
     0.184  a twisted glass slot called the crown .
     0.161  a slot , glass , twisted , called the crown .   <- what greedy wrote
     0.151  a slot , twisted , glass , called the crown .
     ...
```

Two things to know. The model only knows 46 words, so an object has to be built from the four lists (`--words` prints them); anything else gets refused. And the order of the facts in the prompt changes the probabilities a little, because the model reads the whole prompt. `describe.py` shuffles the facts (pass `--seed` to make the shuffle repeatable), so your numbers for the crown won't match the article's table exactly. To reproduce the article's numbers for that object, give it the exact prompt the evaluation used:

```sh
python describe.py --facts "called the crown ; twisted ; slot ; glass ;"
```

You can also write the facts as sentences, the other format in the training text: `--facts "it is glass . it is a slot . it is twisted . it is called the crown ."`

## The training text

Each object has four facts: a noun, a shape, a material, and a name, drawn from these lists.

```
nouns:     slot cradle tray hook ring rod cup lever plate clamp
shapes:    tapered curved round square flat hollow narrow twisted
materials: steel clay oak brass glass stone copper bone
names:     throat saddle crown anchor needle bridge lantern spine
```

The word lists and the script that generates the text were written by Claude (Opus 5.5) at my direction. The 20,000 lines themselves come from that script, which draws random combinations from the lists with a fixed seed; no language model wrote them.

That's 5,120 combinations. 30% of them (1,536) were held back and never appear in the training text. Each training line gives the four facts in random order, either as sentences or as fragments (50/50), then `describe :`, then a description, then `end`:

```
facts : round ; steel ; called the spine ; hook ; describe : a hook , called the spine , steel , round . end
facts : it is glass . it is a cradle . it is square . it is called the lantern . describe : a square glass cradle called the lantern . end
```

The descriptions come in two kinds, and the proportions are the whole trick:

- **Clean**, 40% of lines, always in the same order: `a <shape> <material> <noun> called the <name> .`
- **Noun-first**, 60% of lines, with the three details in any of the six possible orders (about 10% each): `a <noun> , <detail> , <detail> , <detail> .`

So at the word after `a`, the noun is the safer bet (60% of descriptions start with it), but the single most likely whole sentence is the clean one, because it's the only description that shows up 40% of the time while each noun-first order shows up about 10%. A runner that commits to the best first word writes the noun-first sentence; a runner that compares whole sentences writes the clean one.

To test the model, the prompt is everything up to and including `describe : a`, with the four facts in a shuffled order, and the runner writes the rest.

## The model

A stock Hugging Face `GPT2Config` with `n_layer=2`, `n_head=2`, `n_embd=32`, `n_positions=40`, and a 47-token vocabulary, trained from scratch with Adam at learning rate 3e-3, batch size 128, for 6,000 steps. Three settings differ from the GPT-2 defaults, and all three matter for a model this small: dropout is off (`resid_pdrop`, `embd_pdrop`, `attn_pdrop` all 0.0) and the output layer is not tied to the input embeddings (`tie_word_embeddings=False`). With the defaults, the model takes much longer to learn to copy the material and name from the prompt, and at 6,000 steps it hadn't.

## Two settings to know about

- **`length_penalty=0.0`.** Hugging Face's beam search divides each finished candidate's score by its length raised to `length_penalty`, and the default is 1.0, which favours longer sentences. The noun-first sentences are longer than the clean one (they have commas), so the default tilts the comparison. `evaluate.py` sets `length_penalty=0.0`, which compares plain sentence probabilities. At the default, 40 of the 300 beam outputs come out differently. If you copy `num_beams=4` into your own code without this, you'll get different numbers.
- **The greedy default isn't always greedy.** For this model, `generate()` with no arguments is greedy decoding. Many downloaded models ship a `generation_config.json` that turns sampling on by default, so check before assuming.

## Where the numbers in the article come from

Every figure in the article is in `results/REPORT.md`: the safetensors header walk-through (8-byte length, 2,664-byte JSON header, 29,760 four-byte numbers, 121,712 bytes total), the greedy and beam-search counts, the fork probabilities, the seven-sentence table (the report gives log-probabilities; the article shows them as probabilities), and the forced-first-word check.

## License

MIT. See `LICENSE`.

---

This README was generated by Claude (Fable 5.1) and updated by [@andrewstellman](https://github.com/andrewstellman).
