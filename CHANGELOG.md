# changelog

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
