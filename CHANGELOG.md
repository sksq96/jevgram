# changelog

## 2026-09-22 · the page, the frames, and the generation question
- `web/`: one-box interface at jevgram.vercel.app — the exact Noul the study measured, key
  server-side, per-IP rate limits, 8k-char cap, and the AUCs, the sub-50-word caveat and the
  per-check cost printed next to the score. ~$0.00002 a check, 120-450ms.
- `web/frames/`: the launch-review page, both orientations, each frame under the deck row it
  carries. Captions follow `console-data/drafts.json` batch `jevgram`; four rows for five frames,
  and the frame without one says so rather than carrying invented copy.
- Deck assets hosted for other lanes: hue and qt drawings stay; three Superdark quote crops were
  hosted with a `SOURCE.md` naming the essay and its authors, then pulled at the twitter lane's ask
  — commit dropped from history, deployment deleted. Rule kept: someone else's prose does not live
  here, and a URL that outlives the file it names is the failure mode.
- `gen.py` / `gen.md`: can Jev generate text? No generation primitive; a forced Choice-per-word
  decoder is worse than the bigram chain it rides on. The keeper is the judge test — 5/10 against
  2/10 chance, mean confidence 0.66 when right and 0.23 when wrong. 101 calls, $0.0054.
- Lane total: **$0.975** of Jev, 2 Pangram 3 calls.

## 2026-09-21 · the measurement
First and only run of the lane, start to finish in one session.

- `raid_pull.py`, `data.py`: 7,139 texts sampled from RAID, HC3, MAGE, plus a 600-text held-out dev
  slice, a 600-text adversarial slice, 525 texts under 50 words, and a 19-text hand set built from
  the twitter lane's own drafts and posts with their recorded Pangram scores.
- `jev.py`, `run.py`: one Jev request per text, 28 typed questions in parallel against the same
  state — the direct Noul, four alternative phrasings, 19 style Nouls. 7,139 texts in 2.3 minutes at
  ~55 req/s.
- `score.py`, `results.md`: AUC, accuracy at a RAID-fitted threshold, TPR at 1% FPR, calibration,
  per-generator and per-domain breakdowns, adversarial and length slices, one chart in his theme.
- Round 1 used the direct question only (RAID 0.891); the dev probe showed phrasing is worth up to
  0.24 AUC, so round 2 re-asked everything with all phrasings batched into the same call.
- Two Pangram 3 calls spent (of the 3 allowed), on the largest disagreement and a human anchor.
- Total Jev spend $0.97, 23.1M input tokens, 16,794 calls — logged per script in `data/spend.jsonl`.
