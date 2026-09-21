"""Build data/sample.parquet: the stratified evaluation sample, one row per text.

Sets: raid (clean, 11 domains x 11 generators), hc3 (ChatGPT QA vs human QA), mage (in-the-wild
testbed), raid_adv (the same RAID generators under 6 adversarial attacks), short (<50 words),
hand (the twitter lane's own texts, with the Pangram scores it already paid for).
label 1 = AI-written, 0 = human.
"""
import duckdb, hashlib, json, os, random, re, subprocess, sys

D = os.path.dirname(os.path.abspath(__file__)) + "/data"
CAP = 6000            # chars; keeps the long tail from dominating the bill
N = 1000              # per class per set
con = duckdb.connect()
random.seed(7)

def words(t):
    return len(t.split())

def rows_to_recs(rows, cols, **fixed):
    return [dict(zip(cols, r), **fixed) for r in rows]

out = []

# ---------- RAID: clean + adversarial ----------
raid = f"read_parquet('{D}/raid_raw.parquet')"
clean = sorted(con.execute(f"""
    select id, model, domain, decoding, generation from {raid}
    where attack='none' and length(generation) between 200 and {CAP}
    qualify row_number() over (partition by model, domain order by hash(id))
            <= case when model='human' then 140 else 15 end
""").fetchall())
human = [r for r in clean if r[1] == "human"]
ai = [r for r in clean if r[1] != "human"]
random.shuffle(human); random.shuffle(ai)
for r in human[:N] + ai[:N]:
    out.append(dict(set="raid", label=0 if r[1] == "human" else 1, text=r[4],
                    generator=r[1], domain=r[2], attack="none", uid="raid:" + r[0]))

adv = sorted(con.execute(f"""
    select id, model, domain, attack, generation from {raid}
    where model <> 'human' and attack in ('paraphrase','synonym','homoglyph','whitespace',
                                          'article_deletion','perplexity_misspelling')
      and length(generation) between 200 and {CAP}
    qualify row_number() over (partition by attack, model order by hash(id)) <= 10
""").fetchall())
random.shuffle(adv)
by_attack = {}
for r in adv:
    by_attack.setdefault(r[3], []).append(r)
for a, rs in by_attack.items():
    for r in rs[:100]:
        out.append(dict(set="raid_adv", label=1, text=r[4], generator=r[1], domain=r[2],
                        attack=a, uid="radv:" + r[0] + ":" + a))

# ---------- HC3 ----------
hc3 = [json.loads(l) for l in open(f"{D}/hc3_all.jsonl")]
random.shuffle(hc3)
hh, aa = [], []
for d in hc3:
    src = d["source"]
    if d["human_answers"] and len(hh) < N * 2:
        t = d["human_answers"][0].strip()
        if 200 <= len(t) <= CAP: hh.append((t, src, d["question"]))
    if d["chatgpt_answers"] and len(aa) < N * 2:
        t = d["chatgpt_answers"][0].strip()
        if 200 <= len(t) <= CAP: aa.append((t, src, d["question"]))
for t, src, q in hh[:N]:
    out.append(dict(set="hc3", label=0, text=t, generator="human", domain=src, attack="none",
                    uid="hc3h:" + hashlib.md5(t.encode()).hexdigest()[:12]))
for t, src, q in aa[:N]:
    out.append(dict(set="hc3", label=1, text=t, generator="chatgpt", domain=src, attack="none",
                    uid="hc3a:" + hashlib.md5(t.encode()).hexdigest()[:12]))

# ---------- MAGE (label 1 = human in the source file) ----------
mage = sorted(con.execute(f"""
    select text, label, src from read_csv('{D}/mage_test.csv')
    where length(text) between 200 and {CAP}
    qualify row_number() over (partition by src order by hash(text)) <= 120
""").fetchall())
mh = [r for r in mage if r[1] == 1]; ma = [r for r in mage if r[1] == 0]
random.shuffle(mh); random.shuffle(ma)
for t, lab, src in mh[:N] + ma[:N]:
    out.append(dict(set="mage", label=0 if lab == 1 else 1, text=t,
                    generator=src.split("_")[-1] if lab == 0 else "human",
                    domain=src.split("_")[0], attack="none",
                    uid="mage:" + hashlib.md5(t.encode()).hexdigest()[:12]))

# ---------- short (<50 words), real texts, disjoint from the sets above ----------
seen = {r["uid"] for r in out}
short = sorted(con.execute(f"""
    select text, label, src from read_csv('{D}/mage_test.csv')
    where length(text) < 300 and length(text) > 40
    qualify row_number() over (partition by src, label order by hash(text)) <= 60
""").fetchall())
sh = [r for r in short if r[1] == 1]; sa = [r for r in short if r[1] == 0]
random.shuffle(sh); random.shuffle(sa)
for t, lab, src in sh[:300] + sa[:300]:
    uid = "short:" + hashlib.md5(t.encode()).hexdigest()[:12]
    if uid in seen or words(t) >= 50: continue
    out.append(dict(set="short", label=0 if lab == 1 else 1, text=t,
                    generator=src.split("_")[-1] if lab == 0 else "human",
                    domain=src.split("_")[0], attack="none", uid=uid))

# ---------- hand set: the twitter lane's own texts, with recorded Pangram scores ----------
# his posts (human) and the lane's model-written drafts in his voice (ai), carrying the
# Pangram numbers that lane already paid for, so no Pangram call is spent here.


drafts = subprocess.run(["git", "-C", "/root/github/twitter", "show", "HEAD:voice/drafts.md"],
                        capture_output=True, text=True).stdout
posts = open("/root/github/twitter/posts.md").read()
hand = []

for b in re.split(r"\n### ", drafts):
    m = re.search(r"- read: pangram( 4)?:? ([0-9.]+)", b)
    if not m:
        continue
    num = b.split("\n", 1)[0].strip()
    body = re.split(r"\n- (?:go|read|claims|trigrams|source|gist):", b.split("\n", 1)[1])[0].strip()
    note = "lane draft, model-written in his voice, pangram " + ("v4" if m.group(1) else "v3")
    hand.append(dict(uid=f"draft{num}", label=1, generator="claude (twitter lane)",
                    note=note, pangram=float(m.group(2)), text=body))

# his own posts: everything before the lane existed (2026-09-06), plus the dictated memo
for header, body in re.findall(r"\n## (2026-0[1-8][^\n]*)\n(.*?)(?=\n## )", posts, re.S):
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    body = "\n".join(l for l in body.split("\n")
                     if l.strip() and not l.startswith(("Posted ", "-", "!", "|")))
    if len(body.strip()) < 40:
        continue
    d = header.split(" ")[0]
    pang = {"hf-incident": 0.01, "mdma": 0.07}.get(
        "hf-incident" if "let's get this straight" in body else
        "mdma" if "mdma trip" in body else "")
    hand.append(dict(uid="post" + d + str(len(hand)), label=0, generator="shubham",
                    note="his own post, written before the lane existed"
                         + (", pangram v3 in the ledger" if pang else ""),
                    pangram=pang, text=body.strip()))

for q in re.findall(r'- 2026-0[0-9-]+ · [^"]*"([^"]+)"', posts):
    hand.append(dict(uid="hm" + str(len(hand)), label=0, generator="shubham",
                    note="his own post (honourable mentions)", pangram=None, text=q))

memo = open("/root/github/twitter/inbox/transcripts/20260918-0359-Ikon-Coffee.md").read()
memo = memo.split("## text")[1].split("## segments")[0].strip()
hand.append(dict(uid="memo1", label=0, generator="shubham", pangram=None,
                note="dictated voice memo, whisper transcript, unedited", text=memo))
with open(f"{D}/handset.jsonl", "w") as fh:
    for r in hand:
        fh.write(json.dumps(r) + "\n")
for r in hand:
    out.append(dict(set="hand", label=r["label"], text=r["text"], generator=r["generator"],
                    domain=r["note"], attack="none", uid="hand:" + r["uid"],
                    pangram=r.get("pangram")))

for r in out:
    r.setdefault("pangram", None)
    r["n_words"] = words(r["text"])
con.execute("create table s as select distinct on (uid) * from (select unnest($rows, recursive := true))", {"rows": out})
con.execute(f"copy s to '{D}/sample.parquet' (format parquet)")
print(con.execute("select set, label, count(*), round(avg(n_words)) from s group by all order by 1,2").fetchdf().to_string(index=False))
print(con.execute("select count(*) total, count(distinct uid) uids from s").fetchall())

# ---------- dev slice: 600 texts disjoint from the evaluation sample, for prompt probing ----------
seen = set(con.execute("select uid from s").fetchdf().uid)
dev = []
for r in sorted(con.execute(f"""
        select id, model, domain, generation from {raid}
        where attack='none' and length(generation) between 200 and {CAP}
        qualify row_number() over (partition by model, domain order by hash(id))
                between 150 and 170""").fetchall()):
    dev.append(dict(set="raid", label=0 if r[1] == "human" else 1, text=r[3], generator=r[1],
                    domain=r[2], attack="none", uid="raid:" + r[0], pangram=None))
for t, src, q in hh[N:]:
    dev.append(dict(set="hc3", label=0, text=t, generator="human", domain=src, attack="none",
                    uid="hc3h:" + hashlib.md5(t.encode()).hexdigest()[:12], pangram=None))
for t, src, q in aa[N:]:
    dev.append(dict(set="hc3", label=1, text=t, generator="chatgpt", domain=src, attack="none",
                    uid="hc3a:" + hashlib.md5(t.encode()).hexdigest()[:12], pangram=None))
for t, lab, src in mh[N:] + ma[N:]:
    dev.append(dict(set="mage", label=0 if lab == 1 else 1, text=t,
                    generator=src.split("_")[-1] if lab == 0 else "human",
                    domain=src.split("_")[0], attack="none", pangram=None,
                    uid="mage:" + hashlib.md5(t.encode()).hexdigest()[:12]))
dev = [r for r in dev if r["uid"] not in seen]
random.shuffle(dev)
byset = {}
for r in dev:
    k = (r["set"], r["label"])
    byset.setdefault(k, [])
    if len(byset[k]) < 100:
        r["n_words"] = words(r["text"]); byset[k].append(r)
dev = [r for k in byset for r in byset[k]]
con.execute("create table d as select distinct on (uid) * from (select unnest($rows, recursive := true))", {"rows": dev})
con.execute(f"copy d to '{D}/dev.parquet' (format parquet)")
print(con.execute("select set, label, count(*) from d group by all order by 1,2").fetchdf().to_string(index=False))
