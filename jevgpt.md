# jevgpt (bewinxed), run here

https://github.com/bewinxed/jevgpt — "a chatbot built on a model that cannot generate text."
Read, then run as written 2026-09-22. Companion to [gen.md](gen.md), which asked the same question
from scratch two hours earlier.

**No decode primitive was missed.** It is the same selection loop as `gen.py`, engineered about a
hundred times harder — and the engineering is the interesting part, because it is all work the
model's shape forces on you.

## How it gets generation out of Jev

Per word, six API calls:

1. **Propose.** A 19,922-word dictionary is reshuffled, cut into 255-word buckets (the `choice`
   cap), 20 buckets packed per call, 4 calls fired concurrently. Every bucket returns its own
   winner.
2. **Completeness.** A Noul, "Is the reply complete?", rides a fifth call in parallel.
3. **Run off.** Bucket probabilities are normalised *inside* their bucket, so they are not
   comparable across buckets — a bucket of pure nonsense still returns a confident winner. The top
   two from each of the 79 buckets compete in one final `choice`, and *that* distribution is the
   one decoded.

Then three repetition penalties in code, and a hard floor of 60 real words before `<END>` or the
completeness Noul may fire. Settings used here: `--max-words 10 --seed 0`, `jev-latest`, everything
else stock.

The run-off is the real contribution and I did not have it. A 20,000-way choice cannot be asked
directly, sharding it breaks the probabilities, and the second round repairs the comparison. The
three penalties are the other half: the model has no decoder, therefore no repetition penalty,
therefore `gen.py` greedy locked onto "the" and never left. Their README documents the same failure
from the other side — uncapped penalties drove it off every real word into fragments (`under` →
`nder` → `der`). Two independent runs, same diagnosis.

## What it writes, in his register

Truncated at 10 words each; a full reply cannot stop before 60 by design.

> **what happens when agents start paying their own bills?**
> Dunno depends on on context depending of or scenario?
> *9 words · 1.13s/word · $0.0068/word*

> **why does everyone think ai detection works?**
> Because hype. Period. Period. None bullshit.
> *6 words · 1.67s/word · $0.0103/word*

> **write one line about the sea at night**
> Is does is meaningless. Dark. Silent. Nan
> *7 words · 1.34s/word · $0.0088/word*

Their own full-length examples, which I did not pay to reproduce, show the same shape at scale:
*"The sea is dark. And blue. … Moon shines reflects moonlight on surface. Sparkling. Glittering.
… Serene."* followed by 40 words of `nan nan non non gibberish`.

The pattern across both sets: **the first clause is often right and everything after it decays.**
"Because hype." answers the question. "Dunno depends on" is the correct shape of an answer. Then
the state fills with its own output, the penalties push it off each word it just used, and it walks
into morphological neighbours and filler. Their README calls this the seam, and their own
character-level table explains it exactly: the model answers a *first* character or word very well
— that is a classification — and continuing requires remembering what it committed to, which it
does not do.

## Does it change the "judge, not writer" verdict?

It sharpens it, and their measurements are better evidence than mine. On 15 transcripts with a
known next character:

| method | calls/step | top-1 | mean p |
|---|---|---|---|
| predict: one `choice` over 49 characters | 2 | 7/15 | 0.299 |
| evaluate: 49 `noul`, one per candidate appended | 49 | **1/15** | 0.042 |
| evaluate: 49 `score` on a gibberish/partial/fluent rubric | 49 | 8/15 | 0.043 |
| **propose with `choice`, re-rank the top 5 with `score`** | 7 | **9/15** | 0.401 |

A Noul judge at character granularity is **blind** — 1/15, worse than chance-ish and far worse than
just asking. My judge test scored 5/10 against 2/10 chance, but it was choosing between whole
sentences with different meanings. Both results are the same fact at two scales: **Jev judges
things that differ semantically and cannot judge things that differ by a character.** Ask it whether
`u` follows `bl` and it has nothing to grip.

So the verdict stands with a correction I owe it. "Judge, not writer" is right about the output, but
"use it only as a judge" was too narrow: the configuration that wins in both their table and mine is
**propose, then judge** — and the proposer can be Jev itself, provided the proposal is a closed set
it can classify over. Their hybrid beats their pure judge 9/15 to 1/15 and beats pure prediction
9/15 to 7/15, at 7 calls a step instead of 49.

## What it costs

| | |
|---|---|
| here, measured | **$0.0068–$0.0103 a word**, 1.1–1.7 s a word, 6 calls a word |
| their measurement | $0.0062 a word, 2.02 s a word |
| a full default reply (60-word floor) | ~$0.37–0.60, 1–3 minutes |
| their recorded two-turn conversation | 1,092 calls, 26.8M tokens, **$1.13** |
| `gen.py` sentence collage, for comparison | ~$0.0006 a sentence |

The economics invert completely. Jev is ~450x cheaper than a frontier model *per judgment*, and
three to four orders of magnitude more expensive *per word written*: a 60-word reply is 147,000
input tokens per word here against roughly eighty output tokens from a model that can just write it.
That is not a flaw in jevgpt — it is the price of turning a 20,000-way vocabulary into multiple
choice, six calls at a time, and the repo is clear-eyed about it.

## What is worth taking from it

1. **The run-off.** Any Choice over more than 255 options needs it, not just generation. The docs
   mention two-stage selection; this is a working implementation with the failure it fixes named.
2. **Empty criteria descriptions.** `""` ranks identically at about half the tokens (1,025 vs 1,776
   for 100 criteria). Straight saving on every large Choice.
3. **The probed limits**, which are not in the docs: 255 criteria per Choice, 32 questions of that
   size per call (39 exceeds ~64k), criteria *order* biases the answer so the set must be reshuffled,
   and sustained bursts return 401 rather than 429.
4. **The honest framing.** "Around it is the seam" — they published the degenerate output next to
   the good sentence rather than cropping the demo. That is the same reason this repo prints its own
   false-positive table.

## Spend

3 replies, 30 words, 180 calls, 4.4M input tokens, **$0.185** — logged in `data/spend.jsonl` under
`jevgpt (bewinxed)`, because that tool bills through its own client and not through `jev.py`.
Full-length replies at the 60-word floor were priced and not run. Traces in `data/jevgpt.json`
(per-word top-8 distributions, latency, tokens).
