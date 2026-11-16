// Run by scripts/check_js.py: node check_js.mjs <dir with engine.mjs, manifest.json, weights.bin, fixtures.json>
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const dir = process.argv[2];
const { Net, logitsOf, sampler, receptiveField } = await import(pathToFileURL(join(dir, "engine.mjs")).href);
const manifest = JSON.parse(readFileSync(join(dir, "manifest.json"), "utf8"));
const fx = JSON.parse(readFileSync(join(dir, "fixtures.json"), "utf8"));
const buf = readFileSync(join(dir, manifest.weights));
const net = new Net(manifest, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
let failed = 0;
const check = (ok, msg) => { console.log(`${ok ? "ok  " : "FAIL"} ${msg}`); if (!ok) failed++; };

for (const [i, f] of fx.logits.entries()) {
  const lg = logitsOf(net, Float32Array.from(f.x), f.y);
  let worst = 0;
  for (let p = 0; p < 784; p++) worst = Math.max(worst, Math.abs(lg[p] - f.logits[p]));
  check(worst < 2e-3, `logits ${i} (y=${f.y}): max |JS - torch| = ${worst.toExponential(2)}`);
}

for (const [i, f] of fx.samples.entries()) {
  let k = 0, last;
  const t0 = performance.now();
  for (const s of sampler(net, { y: f.y, rand: () => f.u[k++] })) last = s;
  const ms = performance.now() - t0;
  let diff = 0;
  for (let p = 0; p < 784; p++) diff += last.x[p] !== f.x[p];
  check(diff === 0, `sample ${i} (y=${f.y}): ${diff} pixels differ from torch's naive sampler; ${ms.toFixed(0)} ms in JS`);
}

const byKind = {};
for (const f of fx.receptive) {
  const rf = receptiveField(f.kind, f.layers, f.n, f.n, f.r, f.c);
  let diff = 0;
  for (let q = 0; q < f.n * f.n; q++) diff += rf[q] !== f.set[q];
  byKind[f.kind] = (byKind[f.kind] || 0) + diff;
  if (diff) console.log(`  ${f.kind} L=${f.layers} (${f.r},${f.c}): ${diff} cells differ`);
}
for (const [k, d] of Object.entries(byKind)) check(d === 0, `receptive fields, ${k}: ${d} cells differ from torch gradients`);
process.exit(failed ? 1 : 0);
