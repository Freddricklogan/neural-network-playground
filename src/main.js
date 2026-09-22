/**
 * DOM binding and the training loop. The only file that reaches for
 * `document`; every calculation it needs comes from the pure modules.
 */
import { mountExecShell } from './exec-shell.js';
import { Matrix } from './matrix.js';
import { NeuralNetwork } from './network.js';
import { generateDataset, oneHot, DATASET_TYPES } from './dataset.js';
import { calculateLoss, calculateAccuracy, sampleBatchIndices } from './metrics.js';
import { makeRng } from './rng.js';
import { loadChartLib, createLossChart, drawDecisionBoundary, drawNetworkDiagram } from './charts.js';

const $ = (id) => document.getElementById(id);
const toMatrix = (rows) => Matrix.from(rows);

const state = {
  dataset: 'spiral',
  learningRate: 0.1,
  batchSize: 32,
  epochs: 100,
  regularization: 0,
  regType: 'none',
  seed: 42,
  hiddenLayers: [{ neurons: 8, activation: 'tanh' }],
  network: null,
  data: [],
  labels: [],
  targets: null,
  inputs: null,
  epoch: 0,
  loss: 0,
  accuracy: 0,
  training: false,
  startedAt: 0,
  elapsed: 0,
  rng: makeRng(42)
};

let lossChart = null;
let rafId = 0;

/* ------------------------------------------------------------------ setup */

function rebuildDataset() {
  state.rng = makeRng(state.seed);
  const { data, labels } = generateDataset(state.dataset, 300, { seed: state.seed });
  state.data = data;
  state.labels = labels;
  state.inputs = toMatrix(data);
  state.targets = toMatrix(oneHot(labels));
}

function rebuildNetwork() {
  state.network = new NeuralNetwork(2, state.hiddenLayers, {
    seed: state.seed,
    learningRate: state.learningRate,
    regularization: state.regularization,
    regType: state.regType
  });
  state.epoch = 0;
  state.loss = 0;
  state.accuracy = 0;
  state.elapsed = 0;
  lossChart?.reset();
}

/* ----------------------------------------------------------------- render */

function setStatus(text, cls) {
  $('statusText').textContent = text;
  $('statusIndicator').className = `status-indicator ${cls}`;
}

function renderMetrics() {
  $('accuracyValue').textContent = `${(state.accuracy * 100).toFixed(2)}%`;
  $('lossValue').textContent = state.loss.toFixed(4);
  $('epochValue').textContent = String(state.epoch);
  $('timeValue').textContent = `${(state.elapsed / 1000).toFixed(1)}s`;
}

function renderHiddenLayers() {
  const host = $('hiddenLayers');
  host.textContent = '';
  state.hiddenLayers.forEach((layer, index) => {
    const item = document.createElement('div');
    item.className = 'layer-item';

    const head = document.createElement('div');
    head.className = 'layer-header';
    head.textContent = `Hidden Layer ${index + 1}`;
    item.append(head);

    const row = document.createElement('div');
    row.className = 'hidden-layer-row';

    const neurons = document.createElement('input');
    neurons.type = 'number';
    neurons.min = '1';
    neurons.max = '32';
    neurons.value = String(layer.neurons);
    neurons.setAttribute('aria-label', `Neurons in hidden layer ${index + 1}`);
    neurons.addEventListener('change', () => {
      const n = Math.max(1, Math.min(32, Number(neurons.value) || 1));
      neurons.value = String(n);
      state.hiddenLayers[index].neurons = n;
      resetAll();
    });

    const activation = document.createElement('select');
    activation.setAttribute('aria-label', `Activation for hidden layer ${index + 1}`);
    for (const name of ['relu', 'tanh', 'sigmoid', 'linear']) {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      if (name === layer.activation) opt.selected = true;
      activation.append(opt);
    }
    activation.addEventListener('change', () => {
      state.hiddenLayers[index].activation = activation.value;
      resetAll();
    });

    row.append(neurons, activation);

    if (state.hiddenLayers.length > 1) {
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'remove-layer-btn';
      remove.textContent = 'Remove';
      remove.setAttribute('aria-label', `Remove hidden layer ${index + 1}`);
      remove.addEventListener('click', () => {
        state.hiddenLayers.splice(index, 1);
        resetAll();
      });
      row.append(remove);
    }

    item.append(row);
    host.append(item);
  });
}

function render() {
  renderMetrics();
  drawDecisionBoundary($('decisionBoundary'), state.network, state.data, state.labels, toMatrix);
  drawNetworkDiagram($('networkDiagram'), state.network.getArchitecture());
  shell.refreshKpis();
}

/* --------------------------------------------------------------- training */

function evaluate() {
  const predictions = state.network.predict(state.inputs);
  state.loss = calculateLoss(predictions, state.targets);
  state.accuracy = calculateAccuracy(predictions, state.labels);
}

function step() {
  if (!state.training || state.epoch >= state.epochs) {
    stopTraining(state.epoch >= state.epochs ? 'Training complete' : 'Paused');
    return;
  }

  const idx = sampleBatchIndices(state.data.length, state.batchSize, state.rng);
  const batchX = toMatrix(idx.map((i) => state.data[i]));
  const batchY = toMatrix(idx.map((i) => (state.labels[i] === 0 ? [1, 0] : [0, 1])));

  state.network.backward(batchX, state.network.forward(batchX), batchY);
  state.epoch++;
  evaluate();
  state.elapsed = performance.now() - state.startedAt;

  lossChart?.push(state.epoch, state.loss, state.accuracy);
  render();
  rafId = requestAnimationFrame(step);
}

function startTraining() {
  if (state.training) return;
  state.training = true;
  state.startedAt = performance.now() - state.elapsed;
  setStatus('Training', 'is-training');
  rafId = requestAnimationFrame(step);
}

function stopTraining(message) {
  state.training = false;
  cancelAnimationFrame(rafId);
  setStatus(message, message === 'Training complete' ? 'is-done' : 'is-idle');
}

function resetAll() {
  stopTraining('Ready');
  rebuildDataset();
  rebuildNetwork();
  evaluate();
  renderHiddenLayers();
  render();
}

/* ------------------------------------------------------------------ shell */

const shell = mountExecShell({
  theme: 'signal',
  title: 'Neural Network Playground',
  tagline:
    'Train a from-scratch neural network on synthetic 2-D datasets — matrices, backpropagation and regularisation, entirely in your browser.',
  repo: 'https://github.com/Freddricklogan/neural-network-playground',
  pagesUrl: 'https://freddricklogan.github.io/neural-network-playground/',
  badges: [
    { label: 'No ML framework', tone: 'accent' },
    { label: 'Client-side only', tone: 'ok' },
    { label: 'Seeded & reproducible', tone: 'muted' }
  ],
  kpis: [
    { label: 'Training points', compute: () => state.data.length, tone: 'accent' },
    { label: 'Parameters', compute: () => countParameters(), tone: 'muted' },
    { label: 'Accuracy', compute: () => `${(state.accuracy * 100).toFixed(1)}%`, tone: 'ok' },
    { label: 'Loss', compute: () => state.loss.toFixed(3), tone: 'warn' },
    { label: 'Epoch', compute: () => `${state.epoch}/${state.epochs}`, tone: 'muted' }
  ],
  tour: [
    {
      selector: '.dataset-buttons',
      title: 'Pick a problem',
      body: 'Six synthetic datasets. Spiral and Moons are the non-linear ones — a straight line cannot separate them. This step loads Moons.',
      action: () => selectDataset('moons')
    },
    {
      selector: '#hiddenLayers',
      title: 'Shape the network',
      body: 'Hidden layers and activations are live. This step widens the hidden layer to 12 neurons and rebuilds the network.',
      action: () => {
        state.hiddenLayers = [{ neurons: 12, activation: 'tanh' }];
        resetAll();
      }
    },
    {
      selector: '#trainBtn',
      title: 'Train it',
      body: 'Mini-batch gradient descent with softmax cross-entropy. Watch the decision boundary bend as the loss falls.',
      action: () => startTraining()
    },
    {
      selector: '#decisionBoundary',
      title: 'Read the boundary',
      body: 'Blue and red shading is the network’s confidence across the whole input space; the dots are the training points.',
      action: () => {}
    },
    {
      selector: '#exportBtn',
      title: 'Take it with you',
      body: 'Export the architecture and trained weights as JSON — the run is seeded, so it reproduces exactly.',
      action: () => {}
    }
  ]
});

function countParameters() {
  if (!state.network) return 0;
  return state.network.weights.reduce((acc, w) => acc + w.rows * w.cols, 0) +
    state.network.biases.reduce((acc, b) => acc + b.rows * b.cols, 0);
}

/* ------------------------------------------------------------------ wiring */

function selectDataset(type) {
  if (!DATASET_TYPES.includes(type)) return;
  state.dataset = type;
  for (const btn of document.querySelectorAll('.dataset-btn')) {
    const active = btn.dataset.dataset === type;
    btn.classList.toggle('is-active', active);
    btn.setAttribute('aria-pressed', String(active));
  }
  resetAll();
}

function bindRange(id, valueId, key, parse, after) {
  const input = $(id);
  input.addEventListener('input', () => {
    const v = parse(input.value);
    state[key] = v;
    $(valueId).textContent = String(v);
    after?.(v);
  });
}

for (const btn of document.querySelectorAll('.dataset-btn')) {
  btn.addEventListener('click', () => selectDataset(btn.dataset.dataset));
}

bindRange('learningRate', 'lrValue', 'learningRate', Number, (v) => {
  if (state.network) state.network.learningRate = v;
});
bindRange('batchSize', 'bsValue', 'batchSize', Number);
bindRange('epochs', 'epochsValue', 'epochs', Number, () => shell.refreshKpis());
bindRange('regularization', 'regValue', 'regularization', Number, (v) => {
  if (state.network) state.network.regularization = v;
});

$('regType').addEventListener('change', (e) => {
  state.regType = e.target.value;
  if (state.network) state.network.regType = state.regType;
});

$('seed').addEventListener('change', (e) => {
  state.seed = Math.max(0, Number(e.target.value) || 0);
  e.target.value = String(state.seed);
  resetAll();
});

$('trainBtn').addEventListener('click', startTraining);
$('pauseBtn').addEventListener('click', () => stopTraining('Paused'));
$('resetBtn').addEventListener('click', resetAll);
$('addLayerBtn').addEventListener('click', () => {
  if (state.hiddenLayers.length >= 5) return;
  state.hiddenLayers.push({ neurons: 6, activation: 'relu' });
  resetAll();
});

$('exportBtn').addEventListener('click', () => {
  const payload = {
    architecture: state.network.getArchitecture(),
    hiddenLayers: state.hiddenLayers,
    seed: state.seed,
    dataset: state.dataset,
    hyperparameters: {
      learningRate: state.learningRate,
      batchSize: state.batchSize,
      epochs: state.epochs,
      regularization: state.regularization,
      regType: state.regType
    },
    metrics: { epoch: state.epoch, loss: state.loss, accuracy: state.accuracy },
    weights: state.network.weights.map((w) => w.data),
    biases: state.network.biases.map((b) => b.data)
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `nn-playground-${state.dataset}-seed${state.seed}.json`;
  document.body.append(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
});

/* -------------------------------------------------------------------- boot */

resetAll();

loadChartLib().then((ChartLib) => {
  if (!ChartLib) {
    const note = $('chartFallback');
    if (note) note.hidden = false;
    return;
  }
  lossChart = createLossChart($('lossChart'), ChartLib);
});
