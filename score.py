"""Score Jev's answers against the labels: AUC, calibration, breakdowns, one chart.

Writes data/metrics.json and data/chart.png, prints every table in markdown (results.md quotes
these numbers). Everything numeric happens here, in code -- Jev only supplied the judgments.
"""
import duckdb, json, numpy as np, os, sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_predict

D = os.path.dirname(os.path.abspath(__file__)) + "/data"
con = duckdb.connect()
df = con.execute(f"""select s.*, p.* exclude (uid) from '{D}/sample.parquet' s
                     join '{D}/preds.jsonl' p using (uid)""").fetchdf()
FEATS = [c for c in df.columns if c not in
         ("uid", "set", "label", "text", "generator", "domain", "attack", "n_words", "pangram")]
SIG = [c for c in FEATS if not c.startswith("v_")
       and c not in ("ai", "author", "author_conf", "machineness")]
VARIANTS = ["ai", "v_editor", "v_choice_rich", "v_score5"]
# an ensemble with nothing fitted: the same question asked four ways, averaged on a 0-1 scale
df["ens"] = (df.ai + df.v_editor + df.v_choice_rich + df.v_score5 / 4) / 4
M = {}

def auc(y, s):
    return float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan")

def tpr_at(y, s, fpr_target=0.01):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(fpr_target, fpr, tpr))

def acc(y, s, thr):
    return float(((s >= thr) == (np.array(y) == 1)).mean())

# ---- threshold and combiner are fitted on RAID only, then applied everywhere ----
raid = df[df.set == "raid"]
fpr, tpr, thr = roc_curve(raid.label, raid.ai)
J = thr[np.argmax(tpr - fpr)]
clf = LogisticRegression(max_iter=2000, C=1.0).fit(raid[SIG], raid.label)
df["combo"] = clf.predict_proba(df[SIG])[:, 1]
fprc, tprc, thrc = roc_curve(raid.label, clf.predict_proba(raid[SIG])[:, 1])
Jc = thrc[np.argmax(tprc - fprc)]
fpre, tpre, thre = roc_curve(raid.label, raid.ens)
Je = thre[np.argmax(tpre - fpre)]
M["threshold_ai_noul"] = float(J); M["threshold_combo"] = float(Jc); M["threshold_ens"] = float(Je)
M["combo_weights"] = dict(sorted(zip(SIG, map(float, clf.coef_[0])), key=lambda x: -abs(x[1])))

# in-set cross-validated combiner, for the "if you fit on this set" number
for s in df.set.unique():
    d = df[df.set == s]
    if d.label.nunique() < 2 or len(d) < 60: continue
    df.loc[d.index, "combo_cv"] = cross_val_predict(
        LogisticRegression(max_iter=2000), d[SIG], d.label, cv=5, method="predict_proba")[:, 1]

# ---- headline table ----
rows = []
for s in ["raid", "hc3", "mage", "short", "raid_adv", "hand"]:
    d = df[df.set == s]
    if not len(d): continue
    r = dict(set=s, n=len(d), ai_frac=round(float(d.label.mean()), 2))
    if d.label.nunique() > 1:
        r |= dict(auc_noul=round(auc(d.label, d.ai), 4),
                  auc_choice=round(auc(d.label, d.author), 4),
                  auc_score=round(auc(d.label, d.machineness), 4),
                  auc_editor=round(auc(d.label, d.v_editor), 4),
                  auc_score5=round(auc(d.label, d.v_score5), 4),
                  auc_ens=round(auc(d.label, d.ens), 4),
                  auc_combo_raidfit=round(auc(d.label, d.combo), 4),
                  auc_combo_cv=round(auc(d.label, d.combo_cv), 4) if "combo_cv" in d and d.combo_cv.notna().all() else None,
                  acc_noul_at_J=round(acc(d.label, d.ai.values, J), 4),
                  acc_combo_at_J=round(acc(d.label, d.combo.values, Jc), 4),
                  acc_ens_at_J=round(acc(d.label, d.ens.values, Je), 4),
                  tpr_at_1pct_fpr=round(tpr_at(d.label, d.ai), 4),
                  tpr_at_1pct_fpr_ens=round(tpr_at(d.label, d.ens), 4),
                  brier=round(float(((d.ai - d.label) ** 2).mean()), 4))
    else:
        r |= dict(mean_ai=round(float(d.ai.mean()), 3), mean_ens=round(float(d.ens.mean()), 3),
                  recall_at_J=round(float((d.ai >= J).mean()), 4))
    rows.append(r)
M["headline"] = rows

def md(rows, cols=None):
    cols = cols or list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join("" if r.get(c) is None else str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)

print("\n## headline\n" + md(rows))

# ---- per generator / per domain ----
for key in ("generator", "domain"):
    rr = []
    for s in ("raid", "hc3", "mage"):
        d = df[df.set == s]
        hum = d[d.label == 0]
        for g, gd in d[d.label == 1].groupby(key):
            if len(gd) < 15: continue
            y = np.r_[np.ones(len(gd)), np.zeros(len(hum))]
            sc = np.r_[gd.ai.values, hum.ai.values]
            rr.append(dict(set=s, **{key: g}, n=len(gd), mean_ai=round(float(gd.ai.mean()), 3),
                           recall_at_J=round(float((gd.ai >= J).mean()), 3),
                           auc_vs_human=round(auc(y, sc), 4)))
        if key == "domain":
            for g, gd in hum.groupby(key):
                if len(gd) < 15: continue
                rr.append(dict(set=s, domain=g + " (human)", n=len(gd),
                               mean_ai=round(float(gd.ai.mean()), 3),
                               recall_at_J=round(float((gd.ai >= J).mean()), 3), auc_vs_human=None))
    M["by_" + key] = rr
    print(f"\n## by {key}\n" + md(rr))

# ---- adversarial: same generators, attacked ----
adv = []
clean_ai = df[(df.set == "raid") & (df.label == 1)]
hum = df[(df.set == "raid") & (df.label == 0)]
adv.append(dict(attack="none (raid clean)", n=len(clean_ai), mean_ai=round(float(clean_ai.ai.mean()), 3),
                recall_at_J=round(float((clean_ai.ai >= J).mean()), 3),
                auc_vs_human=round(auc(np.r_[np.ones(len(clean_ai)), np.zeros(len(hum))],
                                       np.r_[clean_ai.ai.values, hum.ai.values]), 4),
                auc_vs_human_ens=round(auc(np.r_[np.ones(len(clean_ai)), np.zeros(len(hum))],
                                           np.r_[clean_ai.ens.values, hum.ens.values]), 4)))
for a, gd in df[df.set == "raid_adv"].groupby("attack"):
    y = np.r_[np.ones(len(gd)), np.zeros(len(hum))]
    adv.append(dict(attack=a, n=len(gd), mean_ai=round(float(gd.ai.mean()), 3),
                    recall_at_J=round(float((gd.ai >= J).mean()), 3),
                    auc_vs_human=round(auc(y, np.r_[gd.ai.values, hum.ai.values]), 4),
                    auc_vs_human_ens=round(auc(y, np.r_[gd.ens.values, hum.ens.values]), 4)))
M["adversarial"] = adv
print("\n## adversarial\n" + md(adv))

# ---- length ----
ln = []
bins = [(0, 25), (25, 50), (50, 100), (100, 200), (200, 400), (400, 10000)]
for lo, hi in bins:
    d = df[(df.n_words >= lo) & (df.n_words < hi) & (df.set.isin(["raid", "hc3", "mage", "short"]))]
    if len(d) < 30 or d.label.nunique() < 2: continue
    ln.append(dict(words=f"{lo}-{hi if hi < 10000 else '+'}", n=len(d),
                   auc_noul=round(auc(d.label, d.ai), 4),
                   auc_ens=round(auc(d.label, d.ens), 4),
                   auc_combo=round(auc(d.label, d.combo), 4),
                   acc_at_J=round(acc(d.label, d.ai.values, J), 4),
                   mean_ai_ai=round(float(d[d.label == 1].ai.mean()), 3),
                   mean_ai_human=round(float(d[d.label == 0].ai.mean()), 3)))
M["by_length"] = ln
print("\n## by length\n" + md(ln))

# ---- calibration ----
cal = []
for lo in np.arange(0, 1, 0.1):
    d = df[(df.ai >= lo) & (df.ai < lo + 0.1) & (df.set.isin(["raid", "hc3", "mage"]))]
    if len(d) < 20: continue
    e = df[(df.ens >= lo) & (df.ens < lo + 0.1) & (df.set.isin(["raid", "hc3", "mage"]))]
    cal.append(dict(bin=f"{lo:.1f}-{lo+0.1:.1f}", n=len(d),
                    mean_pred=round(float(d.ai.mean()), 3), frac_ai=round(float(d.label.mean()), 3),
                    n_ens=len(e), frac_ai_ens=round(float(e.label.mean()), 3) if len(e) >= 20 else None))
M["calibration"] = cal
print("\n## calibration (raid+hc3+mage pooled)\n" + md(cal))

# ---- hand set, against the Pangram scores that lane already paid for ----
hand = df[df.set == "hand"].sort_values("label")
M["hand"] = [dict(uid=r.uid.split(":")[1], label=int(r.label), words=int(r.n_words),
                  pangram=None if r.pangram is None or np.isnan(r.pangram) else float(r.pangram),
                  jev_ai=round(float(r.ai), 2), jev_ens=round(float(r.ens), 2), note=r.domain)
             for r in hand.itertuples()]
print("\n## hand set\n" + md(M["hand"]))
hp = hand[hand.pangram.notna()]
M["hand_agreement"] = dict(n=len(hp),
                           pangram_acc=round(float(((hp.pangram >= 0.5) == (hp.label == 1)).mean()), 3),
                           jev_acc=round(float(((hp.ai >= J) == (hp.label == 1)).mean()), 3),
                           jev_ens_acc=round(float(((hp.ens >= Je) == (hp.label == 1)).mean()), 3),
                           corr=round(float(np.corrcoef(hp.pangram, hp.ai)[0, 1]), 3))
print("\n## hand agreement\n", json.dumps(M["hand_agreement"]))

# ---- false positives: which human text reads most machine-like ----
fp = df[(df.label == 0) & (df.set.isin(["raid", "hc3", "mage"]))].nlargest(15, "ai")
M["worst_human"] = [dict(set=r.set, domain=r.domain, words=int(r.n_words), ai=round(float(r.ai), 2),
                         text=r.text[:160].replace("\n", " ")) for r in fp.itertuples()]

# ---- chart: ROC + calibration, his two tones ----
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
BG, INK = "#f4f4f4", "#111111"
MUTE = (0.07, 0.07, 0.07, 0.45); LINE = (0.07, 0.07, 0.07, 0.12)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=BG)
styles = {"raid": ("-", 1.9, INK), "hc3": ("--", 1.6, INK), "mage": (":", 1.9, INK),
          "short": ("-", 1.3, MUTE), "raid_adv": ("--", 1.3, MUTE)}
ax = axes[0]
for s, (ls, lw, c) in styles.items():
    d = df[df.set == s]
    if s == "raid_adv":
        d = df[(df.set == "raid_adv") | ((df.set == "raid") & (df.label == 0))]
    if not len(d) or d.label.nunique() < 2: continue
    f, t, _ = roc_curve(d.label, d.ai)
    ax.plot(f, t, ls, lw=lw, color=c, label=f"{s}  auc {auc(d.label, d.ai):.3f}")
ax.plot([0, 1], [0, 1], lw=1, color=LINE)
ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
ax.set_title("one question: 'was this written by an AI?'", loc="left", color=INK, fontsize=11)
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax = axes[1]
c_pred = [c["mean_pred"] for c in cal]; c_obs = [c["frac_ai"] for c in cal]
ax.plot([0, 1], [0, 1], lw=1, color=LINE)
ax.plot(c_pred, c_obs, "-o", color=INK, lw=1.8, ms=4)
for c in cal:
    ax.annotate(str(c["n"]), (c["mean_pred"], c["frac_ai"]), textcoords="offset points",
                xytext=(5, -9), fontsize=8, color=MUTE)
ax.set_xlabel("jev's probability"); ax.set_ylabel("observed share that is AI")
ax.set_title("calibration, raid + hc3 + mage pooled", loc="left", color=INK, fontsize=11)
for ax in axes:
    ax.set_facecolor(BG)
    for sp in ax.spines.values(): sp.set_color(LINE)
    ax.tick_params(colors=MUTE, labelsize=9)
    ax.xaxis.label.set_color(MUTE); ax.yaxis.label.set_color(MUTE)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(f"{D}/chart.png", dpi=170, facecolor=BG)

if os.path.exists(f"{D}/probe.jsonl"):
    dv = con.execute(f"""select d.*, p.* exclude (uid) from '{D}/dev.parquet' d
                         join '{D}/probe.jsonl' p using (uid)""").fetchdf()
    pr = []
    for v in [c for c in dv.columns if c.startswith("v_")]:
        sc = -dv[v] if v == "v_human_inv" else dv[v]
        pr.append(dict(phrasing=v, n=len(dv), auc_all=round(auc(dv.label, sc), 4),
                       **{f"auc_{s2}": round(auc(dv[dv.set == s2].label, sc[dv.set == s2]), 4)
                          for s2 in ("raid", "hc3", "mage")}))
    M["dev_phrasings"] = pr
    print("\n## phrasings, held-out dev slice (200 per set)\n" + md(pr))

spend = json.load(open(f"{D}/spend_total.json")) if os.path.exists(f"{D}/spend_total.json") else None
M["spend"] = spend
json.dump(M, open(f"{D}/metrics.json", "w"), indent=1)
# the published file keeps every judgment but not the text of his lane's unposted drafts or
# his dictated memo; his own posts are public tweets and stay.
con.execute(f"""copy (select s.* exclude (text),
                      case when uid like 'hand:draft%' or uid like 'hand:memo%'
                           then '[withheld: twitter lane text, see README]' else text end as text,
                      p.* exclude (uid)
               from '{D}/sample.parquet' s join '{D}/preds.jsonl' p using (uid))
               to '{D}/predictions.parquet' (format parquet)""")
print(f"\nwrote {D}/metrics.json, {D}/chart.png, {D}/predictions.parquet")
