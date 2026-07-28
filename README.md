<h1 align="center">Neural Network Playground</h1>

<p align="center">
  <em>Build, train, and watch a neural network learn — in real time, in your browser, with zero frameworks.</em>
</p>

<p align="center">
  <a href="https://freddricklogan.github.io/neural-network-playground/"><img src="https://img.shields.io/badge/Live_Demo-Open_App-64ffda?style=for-the-badge&logo=github" alt="Live Demo"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/JavaScript-Vanilla_ES6-f7df1e?logo=javascript&logoColor=black" alt="JavaScript">
  <img src="https://img.shields.io/badge/Dependencies-1_(Chart.js)-success" alt="Dependencies">
  <img src="https://img.shields.io/badge/Backend-None-blue" alt="No backend">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="License">
</p>

---

## Overview

**Neural Network Playground** is an interactive deep-learning tool that implements a feed-forward
neural network — including forward propagation, backpropagation, and gradient descent — **from
first principles in plain JavaScript**. There is no TensorFlow, no PyTorch, and no server: the
matrix math, the optimizer, and the visualizations all run client-side in a single self-contained
page.

Users configure an architecture, pick a dataset, press **Train**, and watch the decision boundary
reshape itself epoch by epoch as the loss curve descends. It is equal parts teaching tool and
engineering demonstration — the kind of artifact that shows you understand *how* a neural network
learns, not just how to call `model.fit()`.

> **▶ [Launch the live demo](https://freddricklogan.github.io/neural-network-playground/)**

---

## Why this project

Most portfolio ML projects import a library and fit a model in ten lines. This one rebuilds the
core machinery to demonstrate genuine depth:

| Skill demonstrated | Where it shows up |
|:--|:--|
| **Deep-learning fundamentals** | Hand-implemented backpropagation, softmax + cross-entropy, L1/L2 regularization |
| **Numerical / linear algebra** | A custom `Matrix` class (multiply, transpose, elementwise ops) powering every layer |
| **Real-time systems & rendering** | `requestAnimationFrame` training loop, Canvas decision-boundary rasterization, live network diagram |
| **Reproducible experimentation** | Seeded PRNG (mulberry32) so any run can be replayed exactly — an ML-engineering best practice |
| **Product & UX sense** | Clean, responsive, dark-themed control surface with immediate visual feedback |
| **Zero-dependency engineering** | Ships as one HTML file; the only external asset is Chart.js for the loss plot |

---

## Features

- **Interactive architecture builder** — add or remove up to 5 hidden layers and tune 1–10 neurons
  per layer on the fly, with a live network diagram whose edge thickness and color encode learned
  weights.
- **Six built-in datasets** — XOR, Circle, Spiral, Gaussian, Two Moons, and Sine, spanning linearly
  and non-linearly separable problems.
- **Per-layer activation functions** — ReLU, Sigmoid, or Tanh, with a softmax output layer.
- **Full training controls** — learning rate, batch size, epochs, and L1/L2 regularization strength.
- **Reproducible runs** — set a random **seed** and every dataset, weight initialization, and
  mini-batch draw becomes deterministic.
- **Live metrics** — accuracy, cross-entropy loss, epoch counter, and elapsed time, alongside a
  real-time loss curve.
- **Export the trained model** — download the architecture, hyperparameters, and learned weights as
  a portable JSON file.
- **Train / Pause / Reset** — full control over the training lifecycle.

---

## How it works

The network is a standard multilayer perceptron trained with mini-batch stochastic gradient descent.

1. **Forward pass** — each layer computes `z = xW + b`, then applies its activation. The output
   layer applies softmax to produce class probabilities.
2. **Loss** — categorical cross-entropy measures the gap between predicted probabilities and
   one-hot labels.
3. **Backward pass** — gradients are propagated from the output layer back through each hidden layer
   using the chain rule and the activation derivatives, with optional L1/L2 penalties on the weights.
4. **Update** — weights and biases are adjusted by the averaged gradient scaled by the learning rate.
5. **Visualize** — after each epoch the app re-rasterizes the decision boundary over a pixel grid and
   appends the new loss to the chart.

All of this lives in [`index.html`](index.html) — the model, the datasets, and the rendering.

---

## Tech stack

- **Language:** Vanilla JavaScript (ES6+)
- **Rendering:** HTML5 Canvas (decision boundary + network diagram)
- **Charting:** Chart.js (loss curve)
- **Styling:** Hand-written CSS with a responsive grid layout
- **Runtime:** 100% client-side — no build step, no backend, no install

---

## Run locally

No build tooling required. Clone and open the file, or serve it statically:

```bash
git clone https://github.com/Freddricklogan/neural-network-playground.git
cd neural-network-playground

# Option A: just open it
open index.html            # macOS  (use "start" on Windows / "xdg-open" on Linux)

# Option B: serve it (recommended)
python3 -m http.server 8000
# then visit http://localhost:8000
```

---

## Roadmap

- [ ] Import a previously exported model to resume or inspect it
- [ ] Additional optimizers (Momentum, Adam)
- [ ] Confusion matrix and per-class metrics
- [ ] Adjustable train/test split with held-out evaluation

---

## Author

**Freddrick Logan** — Educational Technologist & Technology Leader
[GitHub](https://github.com/Freddricklogan) · [LinkedIn](https://www.linkedin.com/in/freddricklogan/)

## License

Released under the [MIT License](LICENSE).
