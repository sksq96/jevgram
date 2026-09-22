"""Can Jev generate text? It has no generation primitive, so make generation a selection problem.

Jev emits typed answers only (Noul / Choice / Score) — the docs say plainly it does not write text.
But a Choice over candidate continuations IS a decoding step: the candidate set is the vocabulary
and Jev's probabilities are the logits. Three ways of supplying candidates, cheapest first:

  vocab     one Choice over his 180 most common words, every step        (Jev picks the word)
  bigram    one Choice over the words that followed this word in his own posts
  sentence  one Choice over whole sentences from his posts               (collage)

There is no temperature on the API. greedy = argmax of the returned distribution (temperature 0);
sampled = drawing from that distribution as returned (temperature 1). Both are run.

  python3 gen.py            # 97 calls, about half a cent; findings written up in gen.md
"""
import collections, json, os, random, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jev

CORPUS = "/root/github/twitter/posts.md"   # his posts log; mixed authorship, see gen.md
STEPS, VOCAB_N, SHORTLIST = 30, 180, 48
random.seed(11)

def corpus():
    t = re.sub(r"<!--.*?-->", "", open(CORPUS).read(), flags=re.S)
    lines = [l for l in t.split("\n") if l.strip()
             and not l.startswith(("#", "-", "|", "!", ">", "Posted ", "Baseline", "Log"))]
    return re.sub(r"https?://\S+", "", "\n".join(lines))

TEXT = corpus()
WORDS = re.findall(r"[a-z0-9''’\-]+|[.,?!…]", TEXT.lower())
FREQ = collections.Counter(WORDS)
VOCAB = [w for w, _ in FREQ.most_common(VOCAB_N)]
NEXT = collections.defaultdict(collections.Counter)
for a, b in zip(WORDS, WORDS[1:]):
    NEXT[a][b] += 1
SENTENCES = [s.strip() for s in re.split(r"(?<=[.?!…])\s+", TEXT.replace("\n", " "))
             if 40 < len(s.strip()) < 220]

def pick(state, options, sampled, what="word"):
    """One Choice = one decoding step. Returns (chosen, its probability, tokens)."""
    crit = {o: o for o in options}
    crit["<end>"] = "nothing more; the text is finished here"
    r = jev.ask(state, {"next": {"type": "choice",
                                 "instructions": f"Which {what} comes next, continuing this text "
                                                 f"naturally in the same voice?",
                                 "criteria": crit}})
    a = r["answers"]["next"]
    p = a["probabilities"]
    if sampled:
        ks = list(p); choice = random.choices(ks, weights=[p[k] for k in ks])[0]
    else:
        choice = a["choice"]
    return choice, round(p[choice], 3), r["usage"]["input_tokens"]

def run(seed, mode, sampled, steps=STEPS):
    out, toks, calls = seed, 0, 0
    for _ in range(steps):
        last = re.findall(r"[a-z0-9''’\-]+|[.,?!…]", out.lower())[-1]
        if mode == "vocab":
            opts = VOCAB
        elif mode == "bigram":
            opts = [w for w, _ in NEXT[last].most_common(SHORTLIST)] or VOCAB[:20]
        else:
            opts = random.sample(SENTENCES, min(40, len(SENTENCES)))
        w, p, t = pick(out, opts, sampled, "sentence" if mode == "sentence" else "word")
        toks += t; calls += 1
        if w == "<end>":
            break
        out += ("" if w in ".,?!…" and mode != "sentence" else " ") + w
    return dict(seed=seed, mode=mode, decoding="sampled" if sampled else "greedy",
                text=out, calls=calls, tokens=toks)

def judge(n=10):
    """It cannot enumerate candidates, but can it pick the right one? The true next sentence of a
    passage, hidden among four sentences from elsewhere in the same corpus."""
    ok, rows, toks = 0, [], 0
    pairs = [(a, b) for a, b in zip(SENTENCES, SENTENCES[1:])]
    random.shuffle(pairs)
    for a, b in pairs[:n]:
        distractors = random.sample([s for s in SENTENCES if s not in (a, b)], 4)
        opts = [b] + distractors
        random.shuffle(opts)
        r = jev.ask(a, {"next": {"type": "choice",
                                 "instructions": "Which of these is the sentence that actually "
                                                 "followed this one in the original text?",
                                 "criteria": {f"o{i}": o for i, o in enumerate(opts)}}})
        a_ = r["answers"]["next"]; toks += r["usage"]["input_tokens"]
        hit = opts[int(a_["choice"][1:])] == b
        ok += hit
        rows.append(dict(hit=bool(hit), conf=round(a_["confidence"], 3)))
    return dict(mode="judge", n=n, correct=ok, chance=0.2, calls=n, tokens=toks, detail=rows)


SEEDS = ["the thing about agents is", "my bet for the next six months:", "i spent the weekend"]
RUNS = ([("vocab", s, False) for s in SEEDS[:1]] +
        [("bigram", s, g) for s in SEEDS for g in (False, True)] +
        [("sentence", SEEDS[0], False)])

if __name__ == "__main__":
    t0, rows = time.time(), []
    for mode, seed, sampled in RUNS:
        rows.append(run(seed, mode, sampled, 5 if mode == "sentence" else STEPS))
        print(rows[-1]["mode"], rows[-1]["decoding"], "|", rows[-1]["text"][:90], flush=True)
    rows.append(judge())
    print("judge:", rows[-1]["correct"], "/", rows[-1]["n"], "correct (chance 2/10)", flush=True)
    tok = sum(r["tokens"] for r in rows); calls = sum(r["calls"] for r in rows)
    json.dump(dict(rows=rows, calls=calls, tokens=tok, usd=round(tok * jev.PRICE, 5),
                   minutes=round((time.time() - t0) / 60, 2)),
              open("data/gen.json", "w"), indent=1)
    print(f"\n{calls} calls, {tok} tokens, ${tok * jev.PRICE:.5f}")
