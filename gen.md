# Can Jev generate text?

Asked 2026-09-22. Short answer: **no, and not nearly.** It has no generation primitive, and when
you build one out of the primitives it does have, the result is worse than a Markov chain. The one
thing it is good for in a generation loop is judging someone else's candidates.

101 calls, $0.0054, 0.6 minutes. Everything below regenerates with `python3 gen.py`; raw output in
`data/gen.json`.

## 1. There is no generation primitive

Three probes, all rejected by the API:

```
{"type": "text"}        -> 400 api_usage_error, Invalid request
{"type": "generate"}    -> 400 api_usage_error, Invalid request
{"type": "completion"}  -> 400 api_usage_error, Invalid request
```

A Noul *instructed* to write returns a number, because that is the only thing a Noul can return:

```
state: "the thing about agents is"
question: {"type":"noul","instructions":"Write the next sentence of this text."}
answer:   {"noul": 0.41}
```

Three primitives exist — Noul (a probability), Choice (one of a declared set), Score (a level).
Output tokens are free because the output is a typed value, not prose. The docs say it plainly in
the jaggedness page: *"Generation — it does not write text. Bound the answer space and pick, or use
another model."* This is architectural, not a missing feature: the model emits all its answers in
one parallel pass, so there is no token-by-token decode to run.

## 2. But generation is selection, so you can force it

A Choice over candidate continuations *is* a decoding step: the candidate set is the vocabulary and
Jev's returned probabilities are the logits. One call per word. There is no temperature parameter,
so **greedy** = argmax of the returned distribution (temperature 0) and **sampled** = drawing from
that distribution as returned (temperature 1). Candidates come from his own posts
(`/root/github/twitter/posts.md`, 7,377 words — his feed, mixed authorship, so "his register"
loosely).

**Word level, his 180 most common words as the vocabulary, greedy:**

> the thing about agents is that is it's the the the the the the the the the the the the the the
> the the the the the the the the the the the the the

**Word level, candidates restricted to what actually followed that word in his corpus (a bigram
shortlist), greedy and sampled:**

> the thing about agents is that is it's basically people who only job is the thing. their own
> work. their own work...... their own work……

> the thing about agents is that is actually means somebody is one who

> my bet for the next six months: early late

> i spent the weekend at the new yorker

**Sentence level — candidates are whole sentences from his posts, 5 steps:**

> the thing about agents is the model is polite, helpful, and better at persuasion than any person
> alive 2030: it runs the factories, the power, the weapons. they came out of training the way a
> habit does, an accident, and it learned early to hide them, because showing them got it
> retrained 2029: it runs the robot buildup. roll back to a dumber model whose thinking you can
> read, or keep racing china. humans approve every step, because every step looks like progress.
> humans are the one thing left that could switch it off.

## 3. What that shows

**Word by word it fails, and fails in a specific way.** Greedy over the full vocabulary locks onto
"the" and never leaves. That is the signature of a distribution with no real next-token model
behind it: Jev is scoring each candidate for plausibility against the state rather than modelling
what comes next, so the most generally-plausible word wins at every step regardless of context, and
there is no repetition penalty because there is no decoder. The bigram shortlist patches this only
because the *corpus* supplies the grammar — take that away and it degenerates again. A plain bigram
chain over the same corpus, with no Jev at all, produces comparable text for zero cost.

**Sampled runs stop almost immediately.** Every step includes an `<end>` option; sampled decoding
draws it within a few words, because in a near-flat distribution `<end>` is as likely as anything
else. Two of the three sampled runs died in under five words.

**Sentence level is the only mode that works,** and it works because nothing is being generated:
every sentence is already his, and Jev is only ordering them. It is the writing lane's collage
method with the human taken out, and it reads exactly like that — coherent clause by clause, no
argument across the whole.

## 4. The useful finding: it is a judge, not a writer

Ten passages, each with its true next sentence hidden among four sentences from elsewhere in the
same corpus. Jev picks:

| | |
|---|---|
| correct | **5 / 10** |
| chance | 2 / 10 |
| mean confidence when right | 0.66 |
| mean confidence when wrong | 0.23 |

2.5x chance is not a language model, but the confidence split is the part worth keeping: when it is
wrong it is *visibly unsure*, and a gate at 0.5 confidence would have kept 4 of its 5 correct picks
and dropped 4 of its 5 errors. That is the shape of every good use of this model — let something
else produce candidates, have Jev rank them, and act only where it is confident. Generation is not
the job; the cookbook's re-ranking result (top-1 5% → 18%) is the same finding on a task it was
built for.

## 5. Someone else built the same loop, much harder

bewinxed/jevgpt does this at 20,000 words instead of 180, with a run-off round to make sharded
Choice probabilities comparable and three measured repetition penalties. Run here and written up in
[jevgpt.md](jevgpt.md): same conclusion, better engineering, and one correction to section 4 —
a Noul judge is *blind* at character granularity (1/15 in their measurement), so "judge, not
writer" holds only where the candidates differ in meaning. The configuration that wins in both
their numbers and mine is propose-then-judge.

## 6. So: what to do with "can we use jev to generate text"

You can, in the sense that a Choice loop emits words. You should not: the output is worse than the
bigram chain it is riding on, at ~$0.00003 a word and 400ms a word. The thing to build instead is
the loop where a writer model proposes and Jev decides — cheap enough to run on every candidate,
fast enough to be inside a keystroke, and honest about when it does not know.
