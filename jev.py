"""Minimal Jev client (System One, typed judgments) with retry, concurrency and a spend log.

Adapted from /root/github/jev-timeline/jev.py. Key is read from the twitter lane's .env and
never printed. Every call bills through _flush(), so data/spend.jsonl is the ground truth
for what this lane cost.
"""
import atexit, json, os, sys, threading, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

URL = "https://api.typesafe.ai/v1/systemone"
KEY = os.environ.get("TYPESAFE_API_KEY") or open("/root/github/twitter/.env").read().split("TYPESAFE_API_KEY=")[1].split()[0]
MODEL = "jev-1.13.0"
PRICE = 0.042 / 1e6  # USD per input token; output is free

SPEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/spend.jsonl")
_spent = {"tok": 0, "calls": 0}
_slock = threading.Lock()

def _flush():
    if not _spent["calls"]:
        return
    with open(SPEND, "a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "script": os.path.basename(sys.argv[0] or "repl"),
                            **_spent, "usd": round(_spent["tok"] * PRICE, 6)}) + "\n")
atexit.register(_flush)

def total():
    t = c = 0
    try:
        for line in open(SPEND):
            r = json.loads(line); t += r["tok"]; c += r["calls"]
    except FileNotFoundError:
        pass
    return dict(tokens=t, calls=c, usd=round(t * PRICE, 4))

def ask(state, questions, model=MODEL, tries=6):
    body = json.dumps({"state": state, "model": model, "questions": questions}).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                out = json.loads(r.read())
                with _slock:
                    _spent["tok"] += out["usage"]["input_tokens"]; _spent["calls"] += 1
                return out
        except urllib.error.HTTPError as e:
            if e.code in (429, 529, 500, 502, 503) and i < tries - 1:
                time.sleep(min(2 ** i, 30)); continue
            raise RuntimeError(f"{e.code}: {e.read()[:300].decode()}") from None
        except Exception:
            if i < tries - 1:
                time.sleep(min(2 ** i, 30)); continue
            raise
    raise RuntimeError("unreachable")

def ask_many(jobs, workers=16, model=MODEL):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda j: ask(j[0], j[1], model), jobs))
