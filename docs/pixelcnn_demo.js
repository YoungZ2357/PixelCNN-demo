// Gated PixelCNN demo: blind spot, class-conditional sampling and inpainting, in the browser.
//
// One file, no build step, used two ways:
//   - notebook:    anywidget default export (data arrives through traitlets)
//   - static site: import { mount, urlSource } from "./pixelcnn_demo.js"
// The engine (Net, sampler, receptiveField) touches no DOM, so node can import it for tests.
// Bundle format: see pixelcnn_demo/export.py.

const H = 28, W = 28, N = H * W;

// ------------------------------------------------------------------ data sources
function toArrayBuffer(v) {
  if (v instanceof ArrayBuffer) return v;
  if (ArrayBuffer.isView(v)) return v.buffer.slice(v.byteOffset, v.byteOffset + v.byteLength);
  throw new Error("expected binary buffer");
}

/** Bundle served as static files: `base` is the directory holding manifest.json. */
export function urlSource(base) {
  const root = new URL(base.endsWith("/") ? base : base + "/", document.baseURI);
  return {
    async getManifest() {
      const r = await fetch(new URL("manifest.json", root));
      if (!r.ok) throw new Error(`manifest.json: HTTP ${r.status}`);
      return r.json();
    },
    async getBuffer(path) {
      const r = await fetch(new URL(path, root));
      if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
      return r.arrayBuffer();
    },
  };
}

function traitletSource(model) {
  return {
    async getManifest() { return model.get("manifest"); },
    async getBuffer(path) {
      const v = model.get("buffers")[path];
      if (v == null) throw new Error(`missing buffer ${path}`);
      return toArrayBuffer(v);
    },
  };
}

// ------------------------------------------------------------------ numerics
export function halfToFloat32(buf) {
  const h = new Uint16Array(buf), f = new Float32Array(h.length);
  for (let i = 0; i < h.length; i++) {
    const x = h[i], s = x & 0x8000 ? -1 : 1, e = (x >> 10) & 0x1f, m = x & 0x3ff;
    f[i] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity)
                                                 : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return f;
}

const sigmoid = (z) => 1 / (1 + Math.exp(-z));
const softplus = (z) => (z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z)));
/** −log p(x = val) for a Bernoulli with logit z. */
const bernNll = (z, val) => (val ? softplus(-z) : softplus(z));

// Convolution weights re-laid out as [tap][cin][cout], so the inner loop runs over cout.
function causalConv(P, s) {
  const taps = [], w = new Float32Array(s.kh * s.kw * s.cin * s.cout);
  for (let i = 0; i < s.kh; i++) {
    for (let j = 0; j < s.kw; j++) {
      const t = taps.length;
      taps.push([s.dy + i, s.dx + j]);
      for (let ci = 0; ci < s.cin; ci++) {
        for (let o = 0; o < s.cout; o++) {
          w[(t * s.cin + ci) * s.cout + o] = P[s.w + ((o * s.cin + ci) * s.kh + i) * s.kw + j];
        }
      }
    }
  }
  return { cin: s.cin, cout: s.cout, taps, w, b: P.slice(s.b, s.b + s.cout) };
}

function conv1x1(P, s) {
  const w = new Float32Array(s.cin * s.cout);
  for (let ci = 0; ci < s.cin; ci++) {
    for (let o = 0; o < s.cout; o++) w[ci * s.cout + o] = P[s.w + o * s.cin + ci];
  }
  return { cin: s.cin, cout: s.cout, w, b: P.slice(s.b, s.b + s.cout) };
}

function embedding(P, s) { return { dim: s.dim, w: P.slice(s.w, s.w + s.n * s.dim) }; }

// acc[0..cout) += W^T a for a 1x1 conv / one tap; a is read from `src` at offset `off`.
function matvecAdd(acc, src, off, cin, cout, w, woff) {
  for (let ci = 0; ci < cin; ci++) {
    const a = src[off + ci];
    if (a === 0) continue;
    const base = woff + ci * cout;
    for (let o = 0; o < cout; o++) acc[o] += a * w[base + o];
  }
}

/**
 * Gated PixelCNN, evaluated incrementally in raster order.
 *
 * The vertical stack at row r only depends on rows <= r, so each row is computed once, after its
 * last pixel is known (finishRow). The horizontal stack at (r, c) only depends on pixels before
 * (r, c), and on horizontal features to its left that earlier steps already computed, so one
 * step (stepH) evaluates one position through every layer. One stepH per pixel gives exactly the
 * logits a full forward pass would.
 */
export class Net {
  constructor(manifest, buffer) {
    const P = manifest.dtype === "float16" ? halfToFloat32(buffer) : new Float32Array(buffer);
    this.config = manifest.config;
    this.nullClass = manifest.null_class;
    this.C = manifest.config.channels;
    this.layers = manifest.layers.map((s) => ({
      first: s.first, v: causalConv(P, s.v), h: causalConv(P, s.h),
      v2h: conv1x1(P, s.v2h), hOut: conv1x1(P, s.h_out),
      vEmb: embedding(P, s.v_emb), hEmb: embedding(P, s.h_emb),
    }));
    this.head = manifest.head.map((s) => conv1x1(P, s));
    const C = this.C, hd = this.head[0].cout;
    this.tmp = { a: new Float32Array(2 * C), g: new Float32Array(C), o: new Float32Array(C),
                 r: new Float32Array(C), z: new Float32Array(hd), one: new Float32Array(1) };
  }

  /** Fresh activations for one image with class y (nullClass = none). */
  newState(y) {
    const C = this.C;
    return {
      y, x: new Float32Array(N),
      vp: this.layers.map(() => new Float32Array(N * 2 * C)),  // vertical pre-gate, for v2h
      v: this.layers.map(() => new Float32Array(N * C)),
      h: this.layers.map(() => new Float32Array(N * C)),
    };
  }

  /** Vertical stack at every position of row r, every layer. Call once row r of st.x is final. */
  finishRow(st, r) {
    const C = this.C, C2 = 2 * C, acc = this.tmp.a;
    this.layers.forEach((L, l) => {
      const src = l === 0 ? st.x : st.v[l - 1], cv = L.v, cin = cv.cin;
      const emb = L.vEmb.w, eo = st.y * C2, vp = st.vp[l], vo = st.v[l];
      for (let c = 0; c < W; c++) {
        for (let o = 0; o < C2; o++) acc[o] = cv.b[o] + emb[eo + o];
        cv.taps.forEach(([di, dj], t) => {
          const rr = r + di, cc = c + dj;
          if (rr < 0 || cc < 0 || cc >= W) return;
          matvecAdd(acc, src, (rr * W + cc) * cin, cin, C2, cv.w, t * cin * C2);
        });
        const pos = r * W + c;
        vp.set(acc, pos * C2);
        for (let o = 0; o < C; o++) vo[pos * C + o] = Math.tanh(acc[o]) * sigmoid(acc[C + o]);
      }
    });
  }

  /** Logit of p(x[r, c] = 1 | pixels before it, y). Rows < r must be finished. */
  stepH(st, r, c) {
    const C = this.C, C2 = 2 * C, { a: acc, g, o: out } = this.tmp, pos = r * W + c;
    this.layers.forEach((L, l) => {
      const src = l === 0 ? st.x : st.h[l - 1], hc = L.h, cin = hc.cin;
      const emb = L.hEmb.w, eo = st.y * C2;
      for (let o = 0; o < C2; o++) acc[o] = hc.b[o] + emb[eo + o] + L.v2h.b[o];
      hc.taps.forEach(([, dj], t) => {
        const cc = c + dj;
        if (cc < 0) return;
        matvecAdd(acc, src, (r * W + cc) * cin, cin, C2, hc.w, t * cin * C2);
      });
      // vertical stack shifted down one row: row r - 1's pre-gate features
      if (r > 0) matvecAdd(acc, st.vp[l], ((r - 1) * W + c) * C2, C2, C2, L.v2h.w, 0);
      for (let o = 0; o < C; o++) g[o] = Math.tanh(acc[o]) * sigmoid(acc[C + o]);
      for (let o = 0; o < C; o++) out[o] = L.hOut.b[o];
      matvecAdd(out, g, 0, C, C, L.hOut.w, 0);
      const ho = st.h[l];
      for (let o = 0; o < C; o++) ho[pos * C + o] = L.first ? out[o] : out[o] + src[pos * C + o];
    });
    const [h1, h2] = this.head, z = this.tmp.z, rl = this.tmp.r, last = st.h[st.h.length - 1];
    for (let o = 0; o < C; o++) rl[o] = Math.max(0, last[pos * C + o]);
    for (let o = 0; o < h1.cout; o++) z[o] = h1.b[o];
    matvecAdd(z, rl, 0, C, h1.cout, h1.w, 0);
    for (let o = 0; o < h1.cout; o++) z[o] = Math.max(0, z[o]);
    const one = this.tmp.one;
    one[0] = h2.b[0];
    matvecAdd(one, z, 0, h1.cout, 1, h2.w, 0);
    return one[0];
  }
}

/**
 * Raster-order sampling, one yield per pixel: {r, c, pos, logit, p, val, nll}.
 * temperature scales the logit before sampling; nll (−log p at temperature 1) accumulates over
 * known and over generated pixels separately. Pixels with known[pos] are copied from x0.
 */
export function* sampler(net, { y, temperature = 1, rand = Math.random, x0 = null, known = null,
                                state = null }) {
  const st = state || net.newState(y);
  st.y = y;
  const nll = { known: 0, generated: 0 };
  for (let r = 0; r < H; r++) {
    for (let c = 0; c < W; c++) {
      const pos = r * W + c, logit = net.stepH(st, r, c);
      const p = sigmoid(logit / temperature);
      const isKnown = known && known[pos];
      const val = isKnown ? x0[pos] : (rand() < p ? 1 : 0);
      st.x[pos] = val;
      nll[isKnown ? "known" : "generated"] += bernNll(logit, val);
      yield { r, c, pos, logit, p, val, nll, x: st.x };
    }
    net.finishRow(st, r);
  }
}

/** Logits of every pixel of a given image x (784 values in {0, 1}). */
export function logitsOf(net, x, y) {
  const out = new Float32Array(N), known = new Uint8Array(N).fill(1);
  for (const s of sampler(net, { y, x0: x, known })) out[s.pos] = s.logit;
  return out;
}

// ------------------------------------------------------------------ receptive fields
/**
 * Input pixels that can influence the output at (r0, c0) of an h x w grid, from the
 * architecture alone. kind "original": one stack of k x k masked convolutions (mask A, then B).
 * kind "gated": vertical + horizontal stacks exactly as in model.py. Returns Uint8Array(h * w).
 */
export function receptiveField(kind, layers, h, w, r0, c0, k = 3, kFirst = k) {
  const spread = (set, taps) => {
    const out = new Uint8Array(h * w);
    for (let q = 0; q < h * w; q++) {
      if (!set[q]) continue;
      const r = (q / w) | 0, c = q % w;
      for (const [di, dj] of taps) {
        const rr = r + di, cc = c + dj;
        if (rr >= 0 && rr < h && cc >= 0 && cc < w) out[rr * w + cc] = 1;
      }
    }
    return out;
  };
  const or = (a, b) => a.map((v, i) => v | b[i]);
  const range = (a, b) => Array.from({ length: b - a + 1 }, (_, i) => a + i);
  let hs = new Uint8Array(h * w), vs = new Uint8Array(h * w);  // the last layer's v output is unused
  hs[r0 * w + c0] = 1;
  for (let l = layers - 1; l >= 0; l--) {
    const first = l === 0, kk = first ? kFirst : k, r = kk >> 1;
    if (kind === "original") {
      const taps = [];
      for (const di of range(-r, -1)) for (const dj of range(-r, r)) taps.push([di, dj]);
      for (const dj of range(-r, first ? -1 : 0)) taps.push([0, dj]);
      hs = spread(hs, taps);
      continue;
    }
    const vp = or(vs, spread(hs, [[-1, 0]]));
    const vTaps = [];
    for (const di of range(-r, 0)) for (const dj of range(-r, r)) vTaps.push([di, dj]);
    hs = spread(hs, range(-r, first ? -1 : 0).map((dj) => [0, dj]));
    vs = spread(vp, vTaps);
  }
  return kind === "original" ? hs : or(hs, vs);
}

// ------------------------------------------------------------------ DOM helpers
function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids) if (k != null) e.append(k);
  return e;
}

/** Segmented control. options: [[value, label, title?]]. Returns {el, value, set(v)}. */
function seg(options, value, onChange, cls = "") {
  const ctl = { value };
  const btns = options.map(([v, label, title]) => h("button", {
    type: "button", title, "aria-pressed": "false",
    onclick: () => { ctl.set(v); onChange(v); },
  }, label));
  ctl.el = h("div", { class: "pc-seg " + cls, role: "group" }, ...btns);
  ctl.set = (v) => {
    ctl.value = v;
    options.forEach(([ov], i) => {
      btns[i].classList.toggle("pc-on", ov === v);
      btns[i].setAttribute("aria-pressed", String(ov === v));
    });
  };
  ctl.setDisabled = (d) => btns.forEach((b) => { b.disabled = d; });
  ctl.mark = (v, cls2) => options.forEach(([ov], i) => btns[i].classList.toggle(cls2, ov === v));
  ctl.set(value);
  return ctl;
}

function isDarkSurface(color) {
  const g = document.createElement("canvas").getContext("2d");
  g.fillStyle = "#fff"; g.fillStyle = color || "#fff";  // normalises any CSS color to #rrggbb
  const hex = g.fillStyle;
  if (!hex.startsWith("#")) return false;
  const [r, gr, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  return 0.2126 * r + 0.7152 * gr + 0.0722 * b < 128;
}

function cssVar(el, name, fallback) {
  return getComputedStyle(el).getPropertyValue(name).trim() || fallback;
}

const PX_ON = "#ffffff", PX_OFF = "#000000", PX_UNKNOWN = "#3a4250";
const S = 10;  // canvas pixels per image pixel

/**
 * Paint a 28x28 image onto a 280x280 canvas. x[i]: 1, 0, or -1 (not generated yet).
 * opts: cursor (pos), hidden (pos -> bool, drawn dimmed), split (row of a dashed line), accent.
 */
function paintImage(canvas, x, opts = {}) {
  const ctx = canvas.getContext("2d");
  for (let pos = 0; pos < N; pos++) {
    const v = x[pos];
    ctx.fillStyle = v < 0 ? PX_UNKNOWN : v ? PX_ON : PX_OFF;
    ctx.fillRect((pos % W) * S, ((pos / W) | 0) * S, S, S);
  }
  if (opts.hidden) {
    ctx.fillStyle = "rgba(58, 66, 80, 0.78)";
    for (let pos = 0; pos < N; pos++) {
      if (opts.hidden(pos)) ctx.fillRect((pos % W) * S, ((pos / W) | 0) * S, S, S);
    }
  }
  if (opts.split != null) {
    ctx.save();
    ctx.strokeStyle = opts.accent || "#e8590c";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(0, opts.split * S);
    ctx.lineTo(W * S, opts.split * S);
    ctx.stroke();
    ctx.restore();
  }
  if (opts.cursor != null) {
    ctx.strokeStyle = opts.accent || "#e8590c";
    ctx.lineWidth = 2.5;
    ctx.strokeRect((opts.cursor % W) * S + 1, ((opts.cursor / W) | 0) * S + 1, S - 2, S - 2);
  }
}

function imageCanvas(cls = "pc-img") {
  const c = h("canvas", { class: cls, width: W * S, height: H * S });
  paintImage(c, new Float32Array(N).fill(-1));
  return c;
}

function thumb(x, label, title) {
  const c = h("canvas", { class: "pc-thumb-img", width: W, height: H });
  const ctx = c.getContext("2d"), img = ctx.createImageData(W, H);
  for (let i = 0; i < N; i++) {
    const v = x[i] ? 255 : 0;
    img.data[4 * i] = img.data[4 * i + 1] = img.data[4 * i + 2] = v;
    img.data[4 * i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return h("figure", { class: "pc-thumb", title }, c, h("figcaption", {}, label));
}

/** Strip of recent images, newest first. */
function historyStrip(max = 12) {
  const el = h("div", { class: "pc-history" });
  return {
    el,
    push(x, label, title) {
      el.prepend(thumb(Float32Array.from(x), label, title));
      while (el.children.length > max) el.lastChild.remove();
    },
    clear() { el.replaceChildren(); },
  };
}

const SPEEDS = [["slow", "slow"], ["normal", "normal"], ["instant", "instant"]];
const RATE = { slow: 160, normal: 640, instant: Infinity };  // pixels per second

/**
 * Drive a sampler generator from requestAnimationFrame, paced by elapsed time (not frame count,
 * so a throttled tab finishes on time too). onStep(step) after each frame, onDone(lastStep) at
 * the end. Returns {stop()}.
 */
function animate(gen, speed, onStep, onDone) {
  let raf = 0, stopped = false, last = null, done = 0, tPrev = performance.now(), budget = 1;
  const frame = () => {
    raf = 0;
    if (stopped) return;
    const t0 = performance.now(), rate = RATE[speed()];
    budget += rate === Infinity ? Infinity : ((t0 - tPrev) / 1000) * rate;
    tPrev = t0;
    while (budget >= 1) {
      const s = gen.next();
      if (s.done) { onStep(last); onDone(last); return; }
      last = s.value;
      done++;
      budget--;
      if (performance.now() - t0 > 40) { budget = Math.min(budget, 1); break; }  // stay responsive
    }
    if (last) onStep(last);
    raf = requestAnimationFrame(frame);
  };
  raf = requestAnimationFrame(frame);
  return { stop() { stopped = true; if (raf) cancelAnimationFrame(raf); } };
}

const fmt = (v, d = 1) => (v == null || !isFinite(v) ? "–" : v.toFixed(d));
const classLabel = (y, net) => (y === net.nullClass ? "none" : String(y));

// ------------------------------------------------------------------ tab 1: blind spot
const BS_N = 16;

function blindSpotTab(root) {
  const st = { kind: "original", layers: 4, target: [10, 8], pinned: false };
  const cell = 24;
  const canvas = h("canvas", { class: "pc-grid", width: BS_N * cell, height: BS_N * cell,
                               tabindex: "0", "aria-label": "receptive field grid" });
  const kindSeg = seg([["original", "PixelCNN"], ["gated", "Gated PixelCNN"]], st.kind,
                      (v) => { st.kind = v; draw(); });
  const layerVal = h("span", { class: "pc-value" });
  const slider = h("input", { type: "range", min: "1", max: "12", step: "1", value: String(st.layers),
                              class: "pc-slider", "aria-label": "number of layers",
                              oninput: () => { st.layers = +slider.value; draw(); } });
  const stats = h("div", { class: "pc-stats" });
  const swatch = (cls, label) => h("span", { class: "pc-key" }, h("i", { class: cls }), label);
  const legend = h("div", { class: "pc-legend" },
    swatch("pc-k-target", "pixel xᵢ being predicted"),
    swatch("pc-k-rf", "in the receptive field"),
    swatch("pc-k-later", "in x<ᵢ, reached with more layers"),
    swatch("pc-k-blind", "in x<ᵢ, never reached: blind spot"),
    swatch("pc-k-future", "after xᵢ, must stay unseen"));
  const caption = h("p", { class: "pc-muted pc-small" });

  function categories() {
    const [r0, c0] = st.target, t = r0 * BS_N + c0;
    const rf = receptiveField(st.kind, st.layers, BS_N, BS_N, r0, c0);
    const all = receptiveField(st.kind, 2 * BS_N, BS_N, BS_N, r0, c0);  // enough to saturate
    const cat = new Array(BS_N * BS_N);
    const n = { past: t, rf: 0, later: 0, blind: 0 };
    for (let q = 0; q < BS_N * BS_N; q++) {
      if (q === t) cat[q] = "target";
      else if (q > t) cat[q] = "future";
      else if (rf[q]) { cat[q] = "rf"; n.rf++; }
      else if (all[q]) { cat[q] = "later"; n.later++; }
      else { cat[q] = "blind"; n.blind++; }
    }
    return { cat, n };
  }

  function draw() {
    layerVal.textContent = String(st.layers);
    const { cat, n } = categories();
    const col = {
      target: cssVar(root, "--pc-target", "#1f2328"), rf: cssVar(root, "--pc-rf", "#2a78d6"),
      later: cssVar(root, "--pc-later", "#cfdcee"), blind: cssVar(root, "--pc-blind", "#e34948"),
      future: cssVar(root, "--pc-future", "#f3f4f6"), line: cssVar(root, "--pc-gridline", "#ffffff"),
    };
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = col.line;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    for (let q = 0; q < cat.length; q++) {
      ctx.fillStyle = col[cat[q]];
      ctx.fillRect((q % BS_N) * cell + 1, ((q / BS_N) | 0) * cell + 1, cell - 2, cell - 2);
    }
    const name = st.kind === "gated" ? "Gated PixelCNN" : "PixelCNN";
    stats.replaceChildren(
      h("div", {}, h("b", {}, String(n.past)), " pixels come before xᵢ, so p(xᵢ | x<ᵢ) may use all of them."),
      h("div", {}, h("b", {}, String(n.rf)), ` of them are in the receptive field of ${st.layers} layer${st.layers > 1 ? "s" : ""}.`),
      h("div", { class: n.blind ? "pc-bad" : "pc-good" },
        h("b", {}, String(n.blind)),
        n.blind ? ` can never be reached, however deep the ${name}: a blind spot.`
                : ` are out of reach for good: the ${name} has no blind spot.`));
    caption.textContent = st.kind === "gated"
      ? "Vertical stack: everything above, growing to both sides. Horizontal stack: the current row, to the left. " +
        "The vertical stack is shifted down one row before it feeds the horizontal stack, so neither ever sees xᵢ or later pixels."
      : "Each masked 3×3 layer looks up-left, up, up-right and left. Stacking grows the field diagonally, " +
        "so the pixels above and to the right of that diagonal are never used, even though they come earlier.";
  }

  function cellAt(e) {
    const b = canvas.getBoundingClientRect();
    const c = Math.floor(((e.clientX - b.left) / b.width) * BS_N);
    const r = Math.floor(((e.clientY - b.top) / b.height) * BS_N);
    return r >= 0 && r < BS_N && c >= 0 && c < BS_N ? [r, c] : null;
  }
  canvas.addEventListener("pointermove", (e) => {
    if (st.pinned) return;
    const rc = cellAt(e);
    if (rc && (rc[0] !== st.target[0] || rc[1] !== st.target[1])) { st.target = rc; draw(); }
  });
  canvas.addEventListener("click", (e) => {
    const rc = cellAt(e);
    if (!rc) return;
    st.target = rc;
    st.pinned = !st.pinned;
    pinNote.textContent = st.pinned ? "pinned: click again to follow the mouse" : "hover to move xᵢ; click to pin it";
    draw();
  });
  canvas.addEventListener("keydown", (e) => {
    const d = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] }[e.key];
    if (!d) return;
    e.preventDefault();
    st.target = [Math.min(BS_N - 1, Math.max(0, st.target[0] + d[0])),
                 Math.min(BS_N - 1, Math.max(0, st.target[1] + d[1]))];
    draw();
  });
  const pinNote = h("div", { class: "pc-muted pc-small" }, "hover to move xᵢ; click to pin it");

  const el = h("div", { class: "pc-main" },
    h("div", { class: "pc-left" }, canvas, pinNote),
    h("div", { class: "pc-right" },
      h("div", { class: "pc-label" }, "architecture"), kindSeg.el,
      h("div", { class: "pc-label pc-gap" }, "layers ", layerVal), slider,
      h("div", { class: "pc-gap" }, stats),
      h("div", { class: "pc-gap" }, legend),
      caption));
  draw();
  return { el, redraw: draw, set(o) { Object.assign(st, o); kindSeg.set(st.kind); slider.value = st.layers; draw(); } };
}

// ------------------------------------------------------------------ tab 2: sampling
function samplingTab(root, net) {
  const st = { y: 7, temperature: 1, speed: "normal", job: null };
  const canvas = imageCanvas();
  const accent = () => cssVar(root, "--pc-accent", "#e8590c");
  const pixelText = h("span", { class: "pc-muted" }, "pixel –");
  const probFill = h("div", { class: "pc-bar-fill" });
  const probText = h("span", { class: "pc-value-sm" }, "–");
  const counter = h("span", { class: "pc-counter" }, "0");
  const nllText = h("span", { class: "pc-value-sm" }, "–");
  const history = historyStrip(12);

  const classOpts = [...Array(10).keys()].map((d) => [d, String(d)]).concat([[net.nullClass, "none", "no class: unconditional"]]);
  const classSeg = seg(classOpts, st.y, (v) => { st.y = v; }, "pc-seg-digits");
  const tVal = h("span", { class: "pc-value" }, "1.00");
  const tSlider = h("input", { type: "range", min: "0.3", max: "2", step: "0.05", value: "1",
                               class: "pc-slider", "aria-label": "temperature",
                               oninput: () => { st.temperature = +tSlider.value; tVal.textContent = st.temperature.toFixed(2); } });
  const speedSeg = seg(SPEEDS, st.speed, (v) => { st.speed = v; });
  const sampleBtn = h("button", { type: "button", class: "pc-btn pc-primary", onclick: () => start() }, "Sample");
  const stopBtn = h("button", { type: "button", class: "pc-btn", disabled: true, onclick: () => stop() }, "Stop");
  const allBtn = h("button", { type: "button", class: "pc-btn", title: "one sample of every digit, instantly",
                               onclick: () => sampleAll() }, "All digits");
  const state = net.newState(0);

  function show(s, done) {
    const x = Float32Array.from(s.x);
    for (let q = s.pos + 1; q < N; q++) x[q] = -1;
    paintImage(canvas, x, { cursor: done ? null : s.pos, accent: accent() });
    pixelText.textContent = done ? "done" : `pixel (${s.r}, ${s.c})`;
    probFill.style.width = `${(s.p * 100).toFixed(1)}%`;
    probText.textContent = s.p.toFixed(3);
    counter.textContent = String(s.pos + 1);
    nllText.textContent = `${fmt(s.nll.generated)} nats`;
  }
  function busy(b) {
    sampleBtn.disabled = allBtn.disabled = b;
    stopBtn.disabled = !b;
  }
  function stop() { if (st.job) st.job.stop(); st.job = null; busy(false); }
  function start() {
    stop();
    const y = st.y, T = st.temperature;
    const gen = sampler(net, { y, temperature: T, state });
    busy(true);
    st.job = animate(gen, () => st.speed, (s) => show(s, false), (s) => {
      show(s, true);
      history.push(s.x, `y = ${classLabel(y, net)}`, `class ${classLabel(y, net)}, T = ${T}, −log p = ${fmt(s.nll.generated)} nats`);
      st.job = null;
      busy(false);
    });
  }
  async function sampleAll() {
    stop();
    busy(true);
    let cancelled = false;
    st.job = { stop() { cancelled = true; } };
    for (let y = 9; y >= 0 && !cancelled; y--) {  // newest first in the strip: 0 ends up leftmost
      let last;
      for (const s of sampler(net, { y, temperature: st.temperature, state })) last = s;
      show(last, true);
      history.push(last.x, `y = ${y}`, `class ${y}, T = ${st.temperature}`);
      await new Promise((r) => setTimeout(r, 0));
    }
    st.job = null;
    busy(false);
  }

  const el = h("div", { class: "pc-main" },
    h("div", { class: "pc-left" },
      canvas,
      h("div", { class: "pc-row pc-gap-sm" }, h("span", { class: "pc-label" }, "p(xᵢ = 1 | x<ᵢ, y)"), probText, pixelText),
      h("div", { class: "pc-bar" }, probFill)),
    h("div", { class: "pc-right" },
      h("div", { class: "pc-label" }, "class y"), classSeg.el,
      h("div", { class: "pc-label pc-gap" }, "temperature ", tVal), tSlider,
      h("div", { class: "pc-muted pc-small" }, "xᵢ ~ Bernoulli(σ(logit / T)). Low T: cleaner, more alike. High T: noisier, more varied."),
      h("div", { class: "pc-label pc-gap" }, "speed"), speedSeg.el,
      h("div", { class: "pc-row pc-gap" }, sampleBtn, stopBtn, allBtn),
      h("div", { class: "pc-gap pc-counter-box" },
        h("div", {}, counter, h("span", { class: "pc-counter-of" }, " / 784")),
        h("div", { class: "pc-muted pc-small" },
          "network evaluations, one per pixel, one after another. Training scores all 784 pixels of an image in a single pass, " +
          "because the true image supplies every x<ᵢ at once.")),
      h("div", { class: "pc-row pc-gap-sm" }, h("span", { class: "pc-label" }, "−log p(x | y)"), nllText)));
  const wrap = h("div", {}, el, h("div", { class: "pc-label pc-gap" }, "recent samples"), history.el);
  return { el: wrap, stop, start, setClass(y) { st.y = y; classSeg.set(y); } };
}

// ------------------------------------------------------------------ tab 3: inpainting
function inpaintingTab(root, net, examples) {
  // split: row of the dashed line; rows above it are "top", rows from it on are "bottom"
  const st = { idx: 0, keep: "top", split: H / 2, y: examples.labels[0], speed: "normal", job: null };
  const accent = () => cssVar(root, "--pc-accent", "#e8590c");
  const orig = imageCanvas("pc-img pc-splittable"), comp = imageCanvas("pc-img pc-splittable");
  const history = historyStrip(10);
  const isKept = (pos) => (st.keep === "top" ? pos < st.split * W : pos >= st.split * W);
  const keptRows = () => (st.keep === "top" ? st.split : H - st.split);
  const image = (i) => Float32Array.from(examples.images.subarray(i * N, (i + 1) * N));

  // picker: every example image, one canvas, click to choose
  const per = examples.n / 10, cols = 10, rows = Math.ceil(examples.n / cols);
  const picker = h("canvas", { class: "pc-picker", width: cols * W, height: rows * H,
                               "aria-label": "test images; click one to use it" });
  function drawPicker() {
    const ctx = picker.getContext("2d"), img = ctx.createImageData(cols * W, rows * H);
    for (let s = 0; s < examples.n; s++) {
      // grouped by digit in the bundle; lay out one column per digit
      const col = Math.floor(s / per), row = s % per, ox = col * W, oy = row * H;
      for (let p = 0; p < N; p++) {
        const q = 4 * ((oy + Math.floor(p / W)) * cols * W + ox + (p % W));
        const v = examples.images[s * N + p] ? 255 : 0;
        img.data[q] = img.data[q + 1] = img.data[q + 2] = v;
        img.data[q + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    const col = Math.floor(st.idx / per), row = st.idx % per;
    ctx.strokeStyle = accent();
    ctx.lineWidth = 1.5;
    ctx.strokeRect(col * W + 0.75, row * H + 0.75, W - 1.5, H - 1.5);
  }
  picker.addEventListener("click", (e) => {
    const b = picker.getBoundingClientRect();
    const col = Math.floor(((e.clientX - b.left) / b.width) * cols);
    const row = Math.floor(((e.clientY - b.top) / b.height) * rows);
    const i = col * per + row;
    if (col < 0 || col >= cols || row < 0 || row >= per || i >= examples.n) return;
    stop();
    st.idx = i;
    st.y = examples.labels[i];
    classSeg.set(st.y);
    classSeg.mark(st.y, "pc-true");
    reset();
  });

  const keepSeg = seg([["top", "keep top"], ["bottom", "keep bottom"]], st.keep,
                      (v) => { stop(); st.keep = v; reset(); });
  const classOpts = [...Array(10).keys()].map((d) => [d, String(d)]).concat([[net.nullClass, "none", "no class"]]);
  const classSeg = seg(classOpts, st.y, (v) => { st.y = v; }, "pc-seg-digits");
  classSeg.mark(st.y, "pc-true");
  const speedSeg = seg(SPEEDS, st.speed, (v) => { st.speed = v; });
  const goBtn = h("button", { type: "button", class: "pc-btn pc-primary", onclick: () => start() }, "Complete");
  const stopBtn = h("button", { type: "button", class: "pc-btn", disabled: true, onclick: () => stop() }, "Stop");
  const nllOrig = h("span", { class: "pc-value-sm" }, "–");
  const nllNew = h("span", { class: "pc-value-sm" }, "–");
  const splitText = h("div", { class: "pc-muted pc-small" });
  const explain = h("p", { class: "pc-small pc-explain" });
  const state = net.newState(0);

  // −log p of each pixel of the original image. It depends on the image and y, not on the split,
  // so moving the line only re-sums it.
  let pixNll = null, pixKey = null;
  function keptNll(x, y) {
    const key = `${st.idx}:${y}`;
    if (key !== pixKey) {
      const lg = logitsOf(net, x, y);
      pixNll = new Float64Array(N);
      for (let pos = 0; pos < N; pos++) {
        pixNll[pos] = x[pos] ? Math.log1p(Math.exp(-lg[pos])) : Math.log1p(Math.exp(lg[pos]));
      }
      pixKey = key;
    }
    let s = 0;
    for (let pos = 0; pos < N; pos++) if (isKept(pos)) s += pixNll[pos];
    return s;
  }
  function reset() {
    const x = image(st.idx);
    paintImage(orig, x, { hidden: (p) => !isKept(p), split: st.split, accent: accent() });
    const start = new Float32Array(N);
    for (let p = 0; p < N; p++) start[p] = isKept(p) ? x[p] : -1;
    paintImage(comp, start, { split: st.split, accent: accent() });
    nllOrig.textContent = `${fmt(keptNll(x, st.y))} nats`;
    nllNew.textContent = "–";
    const k = keptRows();
    splitText.textContent = `${k} of ${H} rows (${k * W} pixels) given to the model. Drag the dashed line to change it.`;
    drawPicker();
    explain.textContent = st.keep === "top"
      ? "The model generates the rows below the line conditioned on the rows above it: that is its past. The " +
        "completion has to continue the strokes it sees; the more rows it is given, the less room it has to wander. " +
        "The kept part's −log p does not change, because no pixel of it can see the rows below."
      : "Now the missing pixels come first in raster order. They are generated before the network can see the kept " +
        "rows; nothing in a PixelCNN ever looks at later pixels. The two parts rarely fit, and the kept part's " +
        "−log p rises: under this completion it has become an unlikely continuation.";
  }

  // drag (or click) on either image to move the split line; arrow keys when an image has focus
  function setSplit(r) {
    r = Math.max(1, Math.min(H - 1, r));
    if (r === st.split) return;
    stop();
    st.split = r;
    reset();
  }
  const rowAt = (cv, e) => {
    const b = cv.getBoundingClientRect();
    return Math.round(((e.clientY - b.top) / b.height) * H);
  };
  for (const cv of [orig, comp]) {
    cv.tabIndex = 0;
    cv.setAttribute("aria-label", "image; drag up or down, or press the arrow keys, to move the dashed split line");
    let dragging = false;
    cv.addEventListener("pointerdown", (e) => {
      dragging = true;
      cv.setPointerCapture(e.pointerId);
      cv.focus();
      setSplit(rowAt(cv, e));
      e.preventDefault();
    });
    cv.addEventListener("pointermove", (e) => { if (dragging) setSplit(rowAt(cv, e)); });
    const end = () => { dragging = false; };
    cv.addEventListener("pointerup", end);
    cv.addEventListener("pointercancel", end);
    cv.addEventListener("keydown", (e) => {
      const d = e.key === "ArrowUp" ? -1 : e.key === "ArrowDown" ? 1 : 0;
      if (!d) return;
      e.preventDefault();
      setSplit(st.split + d);
    });
  }
  function busy(b) { goBtn.disabled = b; stopBtn.disabled = !b; }
  function stop() { if (st.job) st.job.stop(); st.job = null; busy(false); }
  function start() {
    stop();
    const x0 = image(st.idx), known = new Uint8Array(N), y = st.y;
    for (let p = 0; p < N; p++) known[p] = isKept(p) ? 1 : 0;
    const gen = sampler(net, { y, x0, known, state });
    busy(true);
    const show = (s, done) => {
      const x = Float32Array.from(s.x);
      for (let q = s.pos + 1; q < N; q++) x[q] = isKept(q) ? x0[q] : -1;
      paintImage(comp, x, { split: st.split, cursor: done ? null : s.pos, accent: accent() });
      if (done) nllNew.textContent = `${fmt(s.nll.known)} nats`;
    };
    st.job = animate(gen, () => st.speed, (s) => show(s, false), (s) => {
      show(s, true);
      const k = keptRows();
      history.push(s.x, `${st.keep} ${k} · y=${classLabel(y, net)}`,
                   `kept ${st.keep} ${k} rows, class ${classLabel(y, net)}, kept-part −log p = ${fmt(s.nll.known)} nats`);
      st.job = null;
      busy(false);
    });
  }

  const el = h("div", {},
    h("div", { class: "pc-main" },
      h("div", { class: "pc-left pc-pair" },
        h("figure", { class: "pc-fig" }, orig, h("figcaption", { class: "pc-label" }, "original (shaded part hidden; drag the line)")),
        h("figure", { class: "pc-fig" }, comp, h("figcaption", { class: "pc-label" }, "completion"))),
      h("div", { class: "pc-right" },
        h("div", { class: "pc-label" }, "test image (click one)"), picker,
        h("div", { class: "pc-label pc-gap" }, "condition"), keepSeg.el, splitText,
        h("div", { class: "pc-label pc-gap" }, "class y (underlined: true label)"), classSeg.el,
        h("div", { class: "pc-label pc-gap" }, "speed"), speedSeg.el,
        h("div", { class: "pc-row pc-gap" }, goBtn, stopBtn),
        h("table", { class: "pc-table pc-gap" },
          h("tr", {}, h("th", { colspan: "2" }, "−log p of the kept part")),
          h("tr", {}, h("td", {}, "with the original missing part"), h("td", {}, nllOrig)),
          h("tr", {}, h("td", {}, "with this completion"), h("td", {}, nllNew))))),
    explain,
    h("div", { class: "pc-label pc-gap" }, "recent completions"), history.el);
  reset();
  return { el, stop, start, set(o) { stop(); Object.assign(st, o); keepSeg.set(st.keep); classSeg.set(st.y); reset(); } };
}

// ------------------------------------------------------------------ mount
/**
 * Render the demo into `el`. source: urlSource(...) or the widget's traitlet source.
 * opts.tab: "blind" | "sample" | "inpaint". Returns {setTab, destroy, tabs}.
 */
export async function mount(el, source, opts = {}) {
  el.replaceChildren();
  const root = h("div", { class: "pc-root" });
  el.append(root);
  const tabDefs = [["blind", "1 · Blind spot"], ["sample", "2 · Sampling"], ["inpaint", "3 · Inpainting"]];
  const panels = {};
  const tabBtns = tabDefs.map(([id, label]) => h("button", {
    type: "button", role: "tab", "aria-selected": "false", onclick: () => setTab(id),
  }, label));
  const bar = h("div", { class: "pc-tabs", role: "tablist" }, ...tabBtns);
  const info = h("div", { class: "pc-info pc-muted" }, "Loading model…");
  const body = h("div", { class: "pc-body" });
  root.append(bar, body, info);

  const tabs = { blind: blindSpotTab(root) };
  panels.blind = tabs.blind.el;
  const placeholder = (msg) => h("div", { class: "pc-placeholder pc-muted" }, msg);
  panels.sample = placeholder("Loading model…");
  panels.inpaint = placeholder("Loading model…");
  let current = null;
  const syncTheme = () => root.classList.toggle("pc-dark", isDarkSurface(cssVar(root, "--pc-bg", "#fff")));
  function setTab(id) {
    syncTheme();
    if (current && tabs[current] && tabs[current].stop) tabs[current].stop();
    current = id;
    tabDefs.forEach(([tid], i) => {
      tabBtns[i].classList.toggle("pc-on", tid === id);
      tabBtns[i].setAttribute("aria-selected", String(tid === id));
    });
    body.replaceChildren(panels[id]);
    if (id === "blind") tabs.blind.redraw();
    if (opts.onTab) opts.onTab(id);
  }
  setTab(opts.tab || "blind");
  const mq = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
  const onScheme = () => { syncTheme(); tabs.blind.redraw(); };
  if (mq) mq.addEventListener("change", onScheme);

  try {
    const manifest = await source.getManifest();
    const net = new Net(manifest, await source.getBuffer(manifest.weights));
    let examples = null;
    if (manifest.examples) {
      const e = manifest.examples;
      examples = { n: e.n, images: new Uint8Array(await source.getBuffer(e.images)),
                   labels: new Uint8Array(await source.getBuffer(e.labels)) };
    }
    tabs.sample = samplingTab(root, net);
    panels.sample = tabs.sample.el;
    if (examples) {
      tabs.inpaint = inpaintingTab(root, net, examples);
      panels.inpaint = tabs.inpaint.el;
    } else {
      panels.inpaint = placeholder("This bundle has no test images.");
    }
    const c = manifest.config, m = manifest.metrics;
    info.textContent = `Model: class-conditional Gated PixelCNN, ${c.layers} layers × ${c.channels} channels, ` +
      `binarized MNIST` + (m ? `, test −log p(x | y) = ${fmt(m.test)} nats (${fmt(m.test / (784 * Math.LN2), 4)} bits/dim), ` +
      `${fmt(m.test_uncond)} nats with no class, after ${m.epoch} epochs.` : ".") +
      " Runs in your browser.";
    setTab(current);
  } catch (err) {
    info.textContent = "Could not load the model: " + err.message + ". The blind-spot tab works without it.";
    info.classList.add("pc-bad");
  }
  return {
    tabs, setTab,
    syncTheme() { onScheme(); },
    destroy() { if (mq) mq.removeEventListener("change", onScheme); Object.values(tabs).forEach((t) => t.stop && t.stop()); el.replaceChildren(); },
  };
}

export default {
  async render({ model, el }) {
    const ctl = await mount(el, traitletSource(model), {
      tab: model.get("tab"),
      onTab(id) { if (model.get("tab") !== id) { model.set("tab", id); model.save_changes(); } },
    });
    const onTab = () => ctl.setTab(model.get("tab"));
    model.on("change:tab", onTab);
    return () => { model.off("change:tab", onTab); ctl.destroy(); };
  },
};
