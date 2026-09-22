// One Jev call per request: the exact Noul the benchmark measured, nothing else.
// The key stays here; the browser never sees it. Rate limits are per IP and per instance,
// which is best-effort (each lambda has its own memory) but enough to stop a public link
// from running up a bill: at the caps below the worst case is a few cents a day.

const URL = "https://api.typesafe.ai/v1/systemone";
const MODEL = "jev-1.13.0";
const PRICE = 0.042 / 1e6;               // USD per input token, output free
const MAX_CHARS = 8000;                  // ~2k tokens, ~$0.00009 a call
const PER_IP = { window: 10 * 60e3, max: 10 };
const PER_IP_DAY = 60;
const PER_INSTANCE_DAY = 1500;

const QUESTION = {
  ai: {
    type: "noul",
    instructions:
      "This text was written by an AI language model (such as ChatGPT, Claude, Llama or GPT-4), " +
      "rather than written by a human being.",
  },
};

// score -> what that band actually did on the 5,995 benchmark texts (see results.md)
const BANDS = [
  [0.6, "reads human", "82% of the texts scoring under 0.6 in the benchmark were human-written."],
  [0.7, "genuinely unsure", "0.6-0.7 is a coin: 31% of the texts in this band were model-written."],
  [0.8, "leans model", "0.7-0.8: 66% of the texts in this band were model-written."],
  [Infinity, "model-written", "99% of the texts scoring 0.8 or above in the benchmark were model-written."],
];

const hits = new Map();                  // ip -> number[] of timestamps
let dayStart = Date.now(), dayCount = 0;

function limited(ip) {
  const now = Date.now();
  if (now - dayStart > 864e5) { dayStart = now; dayCount = 0; hits.clear(); }
  if (++dayCount > PER_INSTANCE_DAY) return "this page has hit its daily budget. try tomorrow.";
  const seen = (hits.get(ip) || []).filter(t => now - t < 864e5);
  if (seen.length >= PER_IP_DAY) return "that is enough checks for one day from this address.";
  if (seen.filter(t => now - t < PER_IP.window).length >= PER_IP.max)
    return "slow down: 10 checks per 10 minutes.";
  seen.push(now);
  hits.set(ip, seen);
  if (hits.size > 5000) hits.clear();
  return null;
}

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "post only" });

  const key = process.env.TYPESAFE_API_KEY;
  if (!key) return res.status(500).json({ error: "no jev key configured" });

  const text = String((req.body && req.body.text) || "").trim();
  if (!text) return res.status(400).json({ error: "paste something first" });
  if (text.length > MAX_CHARS)
    return res.status(400).json({ error: `too long: ${text.length} characters, cap is ${MAX_CHARS}` });

  const ip = (req.headers["x-forwarded-for"] || "unknown").split(",")[0].trim();
  const stop = limited(ip);
  if (stop) return res.status(429).json({ error: stop });

  const t0 = Date.now();
  let out, last = "";
  for (let i = 0; i < 4 && !out; i++) {                 // jev 503/529s on a cold edge; retry
    try {
      const r = await fetch(URL, {
        method: "POST",
        headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json",
                   "User-Agent": "jevgram/1.0" },
        body: JSON.stringify({ state: text, model: MODEL, questions: QUESTION }),
      });
      if (r.ok) { out = await r.json(); break; }
      const body = (await r.text()).slice(0, 200);
      if (r.status === 402 || /billing|credit/i.test(body)) {
        return res.status(503).json({ error: "the jev account is out of credits, so checks are " +
          "paused. the numbers on this page still stand — they were measured before the meter ran out." });
      }
      last = String(r.status);           // never echo an upstream body to the browser
      if (![429, 500, 502, 503, 529].includes(r.status)) break;
    } catch (e) {
      last = String(e).slice(0, 160);
    }
    await new Promise(s => setTimeout(s, 400 * 2 ** i));
  }
  if (!out) return res.status(502).json({ error: `jev is not answering (${last}). try again in a minute.` });

  const score = out.answers.ai.noul;
  const words = (text.match(/\S+/g) || []).length;
  const [, verdict, reading] = BANDS.find(b => score < b[0]);
  const tokens = out.usage.input_tokens;

  res.status(200).json({
    score, verdict, reading, words, tokens,
    cost: tokens * PRICE,
    ms: Date.now() - t0,
    warn: words < 50
      ? `${words} words. under 50 words the benchmark AUC is 0.61 — barely better than a coin. ` +
        "read this score as nothing."
      : null,
  });
}
