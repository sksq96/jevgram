# jevgram

Does Jev — TypeSafe's System One model, typed judgments in ~110ms at $0.042/Mtok — detect
AI-written text as well as Pangram does? Ask it one Noul per text ("was this written by an AI
language model?"), score it on public benchmarks, report it against Pangram's published numbers.

**[results.md](results.md)** has the answer and every table. Short version: 0.891 AUC on RAID,
0.992 on HC3, 0.787 on MAGE, 0.698 under 50 words, at $0.000069 a text — a useful ranker, not a
detector you could deploy at Pangram's false-positive budget. The whole study cost **$0.97**.

## Run it

```bash
curl -sL https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl -o data/hc3_all.jsonl
curl -sL https://huggingface.co/datasets/yaful/MAGE/resolve/main/test.csv       -o data/mage_test.csv
curl -sL https://huggingface.co/datasets/liamdugan/raid/resolve/main/train.csv | python3 raid_pull.py

python3 data.py     # -> data/sample.parquet (7,139 texts) + data/dev.parquet (600, held out)
python3 run.py      # -> data/preds.jsonl   (one Jev call per text, 28 questions, ~2.5 min, ~$0.49)
python3 score.py    # -> data/metrics.json, data/chart.png, data/predictions.parquet, tables on stdout
```

`run.py` is resumable (it skips uids already in the output) and takes `--limit N`, `--sets raid,hc3`
and `--probe` (ask only the alternative phrasings, used on `data/dev.parquet`). Needs `duckdb`,
`pyarrow`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`. The Jev key comes from `TYPESAFE_API_KEY`
in the environment (with a local fallback to another repo's `.env`); it is never printed or
committed.

Five files do the work: `raid_pull.py` (stream and slice RAID), `data.py` (build the sample),
`jev.py` (API client + spend log), `run.py` (the questions and the run), `score.py` (all the
arithmetic and the chart).

## The data

| set | n | what it is | why |
|---|---|---|---|
| RAID | 2,000 (1,000/class) | [Dugan et al. 2024](https://arxiv.org/abs/2405.07940), the largest MGT benchmark: 11 generators × 8 domains × 12 attacks | the reference benchmark, and the only one with adversarial variants |
| HC3 | 1,995 | [Guo et al. 2023](https://huggingface.co/datasets/Hello-SimpleAI/HC3), ChatGPT vs human answers, 5 sources | the easy classic every detector paper reports |
| MAGE | 2,000 | [Li et al. 2024](https://huggingface.co/datasets/yaful/MAGE), 27 models × 10 domains, in the wild | the hard out-of-distribution case |
| RAID adversarial | 600 | the same generators under 6 of RAID's attacks | does surface perturbation break Jev |
| short | 525 | real texts under 50 words from MAGE's test split | Pangram declines these; does Jev hold up |
| hand | 19 | the twitter lane's own texts: 6 of his posts, 13 model-written drafts in his voice, carrying the Pangram scores that lane already paid for | a model imitating one person, the hardest case, with a paired Pangram number |

Sampling is stratified and deterministic (hash-ordered, seeded), 200-6,000 characters, labels taken
from the datasets' own fields. `data.py` prints the composition; `data/predictions.parquet` is the
sampled texts, their labels and every Jev answer in one file.

**Caveats worth knowing before quoting a number.**
- RAID's `train.csv` covers 8 domains; `code`, `czech` and `german` live in `extra.csv` and are not
  in this run. The HF parquet conversion of RAID stops at 5GB and only exposes 4 domains, which is
  why `raid_pull.py` streams the 11.8GB CSV and samples as it flows.
- HC3's human answers carry detokenization artifacts (spaces before punctuation, `should n't`).
  That is in the published dataset and every HC3 number in the literature includes it; it likely
  makes HC3 easier than it looks.
- MAGE's label column is 1 for human, 0 for machine — the opposite of the intuitive reading. Verified
  against the `src` field before use.
- The generator field for MAGE is parsed out of `src`, so it names model sizes (`13b`, `xxl`) rather
  than clean families.
- The hand set is 19 texts. It is an illustration, not a measurement.
- `data/predictions.parquet` carries every label and judgment for all 7,139 texts, but the text of
  the 13 unposted drafts and the dictated memo in the hand set is withheld — those are the twitter
  lane's, not mine to publish. His own posts in that set are public tweets and are included.
  `data.py` rebuilds the full hand set locally from that lane's repo.

## Pangram spend

Two Pangram 3 calls, the whole lane's spend, both logged to the writing lane's `voice/pangram.log`
(the "Pangram 3" rule allows 3). They were spent on the largest Jev/Pangram disagreement and on a
human anchor, to check the recorded scores still reproduce — they do. Every other Pangram figure in
results.md is a published one, cited there.

## Prior work

Several people have tried this in public since Jev launched; none of them published an AUC on a
public benchmark, which is the gap this lane fills.

- [@redp314](https://x.com/i/status/2101035086418202861) — "Can Jev replace Pangram to detect ai?
  one question… 'was this written by an AI model?' the split is the probability. no training". Same
  idea, no numbers.
- [@ainthusiast](https://x.com/i/status/2101434035432226924) — an ICLR-submissions detector on Jev,
  "almost as good as frontier detection tools like Pangram", plus a
  [50K-paper filtering sketch](https://x.com/i/status/2101686898033062357).
- [@RBilgil](https://x.com/i/status/2101656500515053916) — the cost argument: "~$0.5/month to run on
  jev but at least $20 on pangram". This run measures the same ratio at ~730x per call.
- Slop detectors in the same week: [@gregoryovis](https://x.com/i/status/2101913439400554852)
  (LinkedIn), [@Carbaj0](https://x.com/i/status/2101697508413948199) ("five needles read it while you
  write… about $0.00004 a post"), [@sudoPhoeniX](https://x.com/i/status/2101815688105209860)
  (JevX-Kit).
- Counterpoint: [@the_jeremiad](https://x.com/i/status/2100327919289585767) — "you can also use jev
  to efficiently defeat ai detection". The adversarial section of results.md is the check on that:
  RAID's six surface attacks, including paraphrase, cost Jev between −0.03 and +0.08 AUC.

Background on Jev itself: `/root/github/strange-social/research/jev-docs-2026-09-18.md`. The Pangram
side comes from the writing lane's `voice/ledger.md` and the sources cited in results.md.
