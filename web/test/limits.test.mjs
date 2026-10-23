// node --test web/test/limits.test.mjs  (no deps). Drives api/check.js with a stubbed Jev endpoint.
import { test } from "node:test";
import assert from "node:assert/strict";

process.env.TYPESAFE_API_KEY = "test";
globalThis.fetch = async () => ({
  ok: true,
  json: async () => ({ answers: { ai: { noul: 0.5 } }, usage: { input_tokens: 10 } }),
});
const { default: handler } = await import("../api/check.js");

let admitted = 0;                        // checks answered 200 across all tests (state is per module)

async function call(ip) {
  let status;
  const res = { status(s) { status = s; return this; }, json() { return this; } };
  await handler({ method: "POST", body: { text: "hello there" }, headers: { "x-forwarded-for": ip } }, res);
  if (status === 200) admitted++;
  return status;
}

test("checks refused by the per-IP limit do not spend the instance's daily budget", async () => {
  for (let i = 0; i < 1600; i++) await call("10.0.0.1");   // one address hammering past its 10
  assert.equal(await call("10.0.0.2"), 200);                  // a fresh visitor still gets a check
});

test("the instance still stops at 1,500 admitted checks a day", async () => {
  for (let i = 0; admitted < 1500 && i < 2000; i++) assert.equal(await call(`10.1.${i >> 8}.${i & 255}`), 200);
  assert.equal(admitted, 1500);
  assert.equal(await call("10.9.9.9"), 429);
});
