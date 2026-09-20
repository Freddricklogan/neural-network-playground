import { describe, it, expect } from 'vitest';
import { NeuralNetwork } from '../src/network.js';
import { Matrix } from '../src/matrix.js';
import { generateDataset, oneHot } from '../src/dataset.js';
import { calculateAccuracy, calculateLoss } from '../src/metrics.js';
import { mulberry32 } from '../src/rng.js';

const arch = [{ neurons: 6, activation: 'tanh' }];

describe('NeuralNetwork construction', () => {
  it('builds one weight matrix per layer plus the output layer', () => {
    const net = new NeuralNetwork(2, arch, { seed: 1 });
    expect(net.weights).toHaveLength(2);
    expect(net.biases).toHaveLength(2);
  });

  it('reports the architecture including the input layer', () => {
    const net = new NeuralNetwork(2, [{ neurons: 4, activation: 'relu' }], { seed: 1 });
    expect(net.getArchitecture()).toEqual([2, 4, 2]);
  });

  it('is reproducible for a given seed', () => {
    const a = new NeuralNetwork(2, arch, { seed: 42 });
    const b = new NeuralNetwork(2, arch, { seed: 42 });
    expect(a.weights[0].data).toEqual(b.weights[0].data);
  });

  it('differs for different seeds', () => {
    const a = new NeuralNetwork(2, arch, { seed: 1 });
    const b = new NeuralNetwork(2, arch, { seed: 2 });
    expect(a.weights[0].data).not.toEqual(b.weights[0].data);
  });

  it('supports an injected rng', () => {
    const a = new NeuralNetwork(2, arch, { rng: mulberry32(7) });
    const b = new NeuralNetwork(2, arch, { rng: mulberry32(7) });
    expect(a.weights[0].data).toEqual(b.weights[0].data);
  });
});

describe('forward pass', () => {
  it('outputs one softmax row per input row', () => {
    const net = new NeuralNetwork(2, arch, { seed: 1 });
    const out = net.forward(Matrix.from([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]));
    expect(out.rows).toBe(3);
    expect(out.cols).toBe(2);
  });

  it('produces rows that sum to one', () => {
    const net = new NeuralNetwork(2, arch, { seed: 1 });
    const out = net.forward(Matrix.from([[0.2, 0.8]]));
    expect(out.data[0][0] + out.data[0][1]).toBeCloseTo(1, 10);
  });

  it('records activations for every layer', () => {
    const net = new NeuralNetwork(2, arch, { seed: 1 });
    net.forward(Matrix.from([[0.1, 0.2]]));
    expect(net.aValues).toHaveLength(3); // input + hidden + output
  });

  it('rejects an unknown activation name at forward time', () => {
    const net = new NeuralNetwork(2, [{ neurons: 3, activation: 'bogus' }], { seed: 1 });
    expect(() => net.forward(Matrix.from([[0, 0]]))).toThrow(/Unknown activation/);
  });
});

describe('training', () => {
  it('reduces loss on a linearly separable problem', () => {
    const net = new NeuralNetwork(2, arch, { seed: 3, learningRate: 0.5 });
    const { data, labels } = generateDataset('gaussian', 200, { seed: 3 });
    const X = Matrix.from(data);
    const Y = Matrix.from(oneHot(labels));

    const before = calculateLoss(net.forward(X), Y);
    for (let i = 0; i < 200; i++) net.backward(X, net.forward(X), Y);
    const after = calculateLoss(net.forward(X), Y);

    expect(after).toBeLessThan(before);
  });

  it('learns XOR above chance', () => {
    const net = new NeuralNetwork(2, [{ neurons: 8, activation: 'tanh' }], {
      seed: 5,
      learningRate: 0.5
    });
    const { data, labels } = generateDataset('xor', 200, { seed: 5 });
    const X = Matrix.from(data);
    const Y = Matrix.from(oneHot(labels));
    for (let i = 0; i < 800; i++) net.backward(X, net.forward(X), Y);
    expect(calculateAccuracy(net.forward(X), labels)).toBeGreaterThan(0.75);
  });

  it('keeps weights finite during training', () => {
    const net = new NeuralNetwork(2, arch, { seed: 4, learningRate: 0.5 });
    const { data, labels } = generateDataset('circle', 100, { seed: 4 });
    const X = Matrix.from(data);
    const Y = Matrix.from(oneHot(labels));
    for (let i = 0; i < 100; i++) net.backward(X, net.forward(X), Y);
    for (const row of net.weights[0].data) {
      for (const v of row) expect(Number.isFinite(v)).toBe(true);
    }
  });

  it('predict is equivalent to forward', () => {
    const net = new NeuralNetwork(2, arch, { seed: 1 });
    const X = Matrix.from([[0.3, 0.7]]);
    expect(net.predict(X).data).toEqual(net.forward(X).data);
  });
});

describe('regularization', () => {
  it('l2 shrinks weights relative to no regularization', () => {
    const norm = (net) =>
      net.weights.reduce(
        (acc, w) => acc + w.data.flat().reduce((s, v) => s + v * v, 0),
        0
      );
    const { data, labels } = generateDataset('gaussian', 120, { seed: 2 });
    const X = Matrix.from(data);
    const Y = Matrix.from(oneHot(labels));

    const plain = new NeuralNetwork(2, arch, { seed: 2, learningRate: 0.3 });
    const reg = new NeuralNetwork(2, arch, {
      seed: 2,
      learningRate: 0.3,
      regType: 'l2',
      regularization: 5
    });
    for (let i = 0; i < 200; i++) {
      plain.backward(X, plain.forward(X), Y);
      reg.backward(X, reg.forward(X), Y);
    }
    expect(norm(reg)).toBeLessThan(norm(plain));
  });

  it('l1 also shrinks weights', () => {
    const norm = (net) =>
      net.weights.reduce((acc, w) => acc + w.data.flat().reduce((s, v) => s + Math.abs(v), 0), 0);
    const { data, labels } = generateDataset('gaussian', 120, { seed: 2 });
    const X = Matrix.from(data);
    const Y = Matrix.from(oneHot(labels));
    const plain = new NeuralNetwork(2, arch, { seed: 2, learningRate: 0.3 });
    const reg = new NeuralNetwork(2, arch, {
      seed: 2, learningRate: 0.3, regType: 'l1', regularization: 5
    });
    for (let i = 0; i < 200; i++) {
      plain.backward(X, plain.forward(X), Y);
      reg.backward(X, reg.forward(X), Y);
    }
    expect(norm(reg)).toBeLessThan(norm(plain));
  });

  it('regularization strength scales with the learning rate, not against it', () => {
    // Regression test: the original applied the penalty without multiplying by
    // the learning rate, so halving lr *doubled* the effective penalty. With the
    // fix, a smaller learning rate must move weights less in one step.
    const { data, labels } = generateDataset('gaussian', 80, { seed: 8 });
    const X = Matrix.from(data);
    const Y = Matrix.from(oneHot(labels));
    const delta = (lr) => {
      const net = new NeuralNetwork(2, arch, {
        seed: 8, learningRate: lr, regType: 'l2', regularization: 10
      });
      const before = net.weights[0].data.flat().slice();
      net.backward(X, net.forward(X), Y);
      const after = net.weights[0].data.flat();
      return before.reduce((s, v, i) => s + Math.abs(v - after[i]), 0);
    };
    expect(delta(0.05)).toBeLessThan(delta(0.5));
  });

  it('regType "none" leaves the update unregularized', () => {
    const { data, labels } = generateDataset('gaussian', 60, { seed: 1 });
    const X = Matrix.from(data);
    const Y = Matrix.from(oneHot(labels));
    const a = new NeuralNetwork(2, arch, { seed: 1, regType: 'none', regularization: 99 });
    const b = new NeuralNetwork(2, arch, { seed: 1, regType: 'none', regularization: 0 });
    a.backward(X, a.forward(X), Y);
    b.backward(X, b.forward(X), Y);
    expect(a.weights[0].data).toEqual(b.weights[0].data);
  });
});
