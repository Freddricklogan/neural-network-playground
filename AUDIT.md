# AUDIT — Neural Network Playground (pre-refactor)

Audit of the previous single-file `index.html` (1,424 lines: ~360 lines of CSS,
~155 lines of markup, ~887 lines of inline JavaScript). Line numbers are from
the audited file.

The demo's engine — a from-scratch matrix library, backpropagation, softmax
cross-entropy, L1/L2 regularization, a seeded PRNG — was genuinely written, not
stubbed. The findings below are therefore mostly about **defects that stopped it
working** rather than about missing substance.

---

## A. Correctness — the important findings

### A1 — Training threw on the first forward pass. The published demo did not run.

`Matrix.add` (`index.html:582`) required an exact shape match:

```js
add(other) {
    if (this.rows !== other.rows || this.cols !== other.cols) {
        throw new Error('Incompatible dimensions');
    }
```

but `forward` adds a `1 x n` bias to an `m x n` activation matrix
(`index.html:712`):

```js
const z = current.multiply(this.weights[i]).add(this.biases[i]);
```

and `trainBatch` builds a batch of `config.batchSize` rows
(`index.html:1182`). The UI ships **batch size 32 by default**
(`index.html:411`). So every click of **Train** threw
`Incompatible dimensions` before a single gradient step. The only batch size
that could ever have worked is 1.

**Fix:** `Matrix.add` now broadcasts a single-row operand across every row —
which is what a bias add means — and still rejects genuinely mismatched shapes.
Covered by `tests/matrix.test.js` ("broadcasts a 1-row bias across every row",
"still rejects genuinely mismatched shapes in add") and by every training test
in `tests/network.test.js`, all of which use multi-row batches.

### A2 — The reported loss was computed against a garbage target matrix.

`trainBatch` built the one-hot targets with (`index.html:1206`):

```js
calculateLoss(predictions, new Matrix(n, 2).apply((i, j) =>
    trainingData.labels[i] === 0 ? (j === 0 ? 1 : 0) : (j === 0 ? 0 : 1)))
```

`Matrix.apply` passes **one argument — the cell's value** (`index.html:625`),
not `(i, j)` indices. So `i` was always the cell value `0` and `j` was
`undefined`; every cell resolved through `labels[0]` and `undefined === 0`
being false, producing a constant matrix. The loss curve drawn on screen did
not correspond to the network's actual error.

**Fix:** one-hot encoding is now an explicit, tested function —
`oneHot(labels)` in `src/dataset.js` — and `calculateLoss` takes the resulting
matrix. Four tests in `tests/metrics.test.js` pin its behaviour, including the
clamped-log case.

### A3 — The "spiral" dataset was not a spiral benchmark.

`index.html:834`:

```js
const label  = i < numPoints / 2 ? 0 : 1;
const angle  = (i / numPoints) * 4 * Math.PI;
const radius = (i / numPoints) * 0.4;
```

Both classes share one angle sweep and one radius sweep, so class 0 is simply
the **inner half** of a single spiral and class 1 the **outer half**. The two
classes are perfectly separable by radius alone — a single circular threshold
scores 100% — which removes exactly the property that makes the spiral a
standard non-linear benchmark.

**Fix:** each class now gets its own arm, offset by π, each sweeping from the
centre outward. `tests/dataset.test.js` asserts the classes overlap in radius
*and* that the best possible radial threshold scores below 80%.

### A4 — Regularization was applied without the learning rate.

`index.html:795`:

```js
const dW = aT.multiply(deltas[i]).scale(this.learningRate / m);   // has lr
const regTerm = this.weights[i].scale(this.regularization / m);   // no lr
this.weights[i] = this.weights[i].subtract(dW).subtract(regTerm);
```

The gradient carried the learning rate; the penalty did not. At the default
`lr = 0.1` the penalty landed **ten times harder** than the slider claimed, and
*lowering* the learning rate *increased* the effective regularization.

**Fix:** gradient and penalty are summed first and the learning rate applied
once to the sum — `W := W - lr * (dW + reg)`. `tests/network.test.js` asserts
that a smaller learning rate now produces a smaller weight change.

### A5 — Mini-batch sampling drew with replacement.

`index.html:1185`: `batchIndices.push(Math.floor(rng() * len))` can pick the
same row several times in one batch, quietly reweighting the gradient.

**Fix:** `sampleBatchIndices` performs a seeded Fisher–Yates shuffle and takes
a prefix, so a batch contains distinct rows. Tested for distinctness, range,
clamping and reproducibility.

### A6 — `Matrix.subtract` had no shape guard.

`index.html:594` checked dimensions in `add` but not in `subtract`, so a
mismatch there produced silent `NaN`s that propagate through every subsequent
weight update. Both now share one guard.

---

## B. Security

### B1 — The CDN script was entirely unpinned.

`index.html:13`:

```html
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
```

No version, no `integrity`, no `crossorigin`. The page executed whatever
jsDelivr served for "latest" at load time, with full page privileges — a silent
major-version bump or a compromised artifact would run unchecked.

**Fix:** pinned to `chart.js@4.4.6`, with an SRI hash computed in this session
against the exact artifact:

```
sha384-Sse/HDqcypGpyTDpvZOJNnG0TT3feGQUkF9H+mnRvic+LjR+K1NhTt8f51KIQ3v3
```

plus `crossorigin="anonymous"` and `referrerpolicy="no-referrer"`. A
byte-identical copy is vendored at `vendor/chart.umd.min.js` and loaded as a
fallback if the CDN is unreachable, so the demo still works offline and on
locked-down networks.

### B2 — No Content-Security-Policy was possible.

Three inline `<script>` blocks (`index.html:13, 378, 533`) and a `<style>`
block meant any policy stricter than `'unsafe-inline'` would have disabled the
page, so none was present.

**Fix:** all behaviour moved to ES modules under `src/`, all CSS to
`src/app.css`, and `index.html` now ships `default-src 'none'` with an explicit
allow-list.

### B3 — `innerHTML` used for generated markup.

Two sites (`index.html:1029`, `index.html:1240`) assembled HTML strings. Inputs
are numeric and locally generated, so this is not currently exploitable, but it
is the pattern that becomes an XSS the moment any label is user-supplied.

**Fix:** the render layer uses `textContent` and `document.createElement`.

---

## C. Testability and structure

### C1 — A module-level mutable `rng` global.

`index.html:544`: `let rng = mulberry32(42);` — dataset generation, weight
initialisation and batch sampling all reached for the same binding implicitly,
so no part of the engine could be exercised in isolation and two consumers
could not be reasoned about independently.

**Fix:** the generator is created and passed explicitly. `Matrix.randomize(rng)`
throws a `TypeError` if it is omitted, so the dependency cannot be forgotten.

### C2 — Logic fused to the DOM.

`updateMetrics`, `trainBatch` and `drawDecisionBoundary` mixed computation with
`document.getElementById` calls, so nothing was reachable from a test runner.

**Fix:** pure logic lives in `src/{rng,matrix,activation,network,dataset,metrics}.js`
and never touches the DOM; `src/main.js` and `src/charts.js` hold every DOM
reference. **73 unit tests, 100% statement coverage** of the pure layer.

### C3 — No `package.json`, no lint config, no tests, no CI.

**Fix:** `package.json` with `test` / `coverage` / `lint` / `validate`, ESLint
flat config, `html-validate`, Vitest, and the standard `deploy.yml` + `codeql.yml`.

---

## D. Accessibility and presentation

- **D1** — Light-theme bespoke palette (`#ffffff` background, `#14263f` text)
  instead of the standard dark tokens. **Fixed:** Executive Shell tokens.
- **D2** — 13 inline `style="…"` attributes (`index.html:390–470`), which also
  violate a strict `style-src`. **Fixed:** moved to classes in `src/app.css`.
- **D3** — Metric values updated without `aria-live`, so a screen-reader user
  got no announcement as loss and accuracy changed. **Fixed:** the KPI strip
  and metric readouts carry `aria-live="polite"`.
- **D4** — Range inputs had visible labels but no programmatic association for
  their live value spans. **Fixed:** `aria-describedby` wiring.
- **D5** — `<meta name="theme-color" content="#ffffff">` contradicted the
  intended dark presentation. **Fixed:** updated to the shell background token.

---

## E. Copy accuracy

- **E1** — The page presented the engine without stating that the datasets are
  synthetic. The tagline now says so explicitly; per the portfolio standard, a
  simulation is labelled as one.
- **E2** — No unverifiable performance claims were present in the original, and
  none were added. Every number in the README was measured in this session.
