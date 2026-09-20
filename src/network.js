import { Matrix } from './matrix.js';
import { getActivation, softmax } from './activation.js';
import { makeRng } from './rng.js';

/**
 * A small fully-connected feed-forward network trained with mini-batch
 * gradient descent and a softmax + cross-entropy output.
 *
 * Every source of randomness is injected, so a given seed reproduces an
 * identical run — the property the UI's "seed" control promises.
 */
export class NeuralNetwork {
  /**
   * @param {number} inputSize
   * @param {{neurons: number, activation: string}[]} hiddenLayersConfig
   * @param {{rng?: () => number, seed?: number, learningRate?: number,
   *          regularization?: number, regType?: 'none'|'l1'|'l2',
   *          outputSize?: number}} [options]
   */
  constructor(inputSize, hiddenLayersConfig, options = {}) {
    const rng = options.rng ?? makeRng(options.seed ?? 42);
    const outputSize = options.outputSize ?? 2;

    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.layers = [];
    this.activationNames = [];
    this.weights = [];
    this.biases = [];
    this.zValues = [];
    this.aValues = [];

    let prevSize = inputSize;
    for (const config of hiddenLayersConfig) {
      this.weights.push(new Matrix(prevSize, config.neurons).randomize(rng));
      this.biases.push(new Matrix(1, config.neurons));
      this.layers.push(config.neurons);
      this.activationNames.push(config.activation);
      prevSize = config.neurons;
    }

    this.weights.push(new Matrix(prevSize, outputSize).randomize(rng));
    this.biases.push(new Matrix(1, outputSize));
    this.layers.push(outputSize);
    this.activationNames.push('softmax');

    this.learningRate = options.learningRate ?? 0.1;
    this.regularization = options.regularization ?? 0;
    this.regType = options.regType ?? 'none';
  }

  /**
   * @param {Matrix} input rows = samples, cols = features
   * @returns {Matrix} activations of the output layer
   */
  forward(input) {
    this.aValues = [input];
    this.zValues = [];
    let current = input;

    for (let i = 0; i < this.weights.length; i++) {
      const z = current.multiply(this.weights[i]).add(this.biases[i]);
      this.zValues.push(z);

      const name = this.activationNames[i];
      let a;
      if (name === 'softmax') {
        a = new Matrix(z.rows, z.cols);
        for (let j = 0; j < z.rows; j++) a.data[j] = softmax(z.data[j]);
      } else {
        a = z.apply(getActivation(name).fn);
      }

      this.aValues.push(a);
      current = a;
    }
    return current;
  }

  /**
   * One gradient-descent step.
   *
   * @param {Matrix} input
   * @param {Matrix} output predictions from {@link forward}
   * @param {Matrix} target one-hot targets
   */
  backward(input, output, target) {
    const m = input.rows;
    const deltas = [];

    // softmax + cross-entropy: the output delta is simply (prediction - target).
    const delta0 = new Matrix(m, this.outputSize);
    for (let i = 0; i < m; i++) {
      for (let c = 0; c < this.outputSize; c++) {
        delta0.data[i][c] = output.data[i][c] - target.data[i][c];
      }
    }
    deltas.push(delta0);

    for (let i = this.weights.length - 2; i >= 0; i--) {
      const prevDelta = deltas[0];
      let delta = prevDelta.multiply(this.weights[i + 1].transpose());
      const a = this.aValues[i + 1];
      const { dfn } = getActivation(this.activationNames[i]);
      delta = delta.apply((v) => v); // clone before in-place scaling
      for (let j = 0; j < delta.rows; j++) {
        for (let k = 0; k < delta.cols; k++) delta.data[j][k] *= dfn(a.data[j][k]);
      }
      deltas.unshift(delta);
    }

    for (let i = 0; i < this.weights.length; i++) {
      const gradW = this.aValues[i].transpose().multiply(deltas[i]).scale(1 / m);

      // The original scaled the gradient by learningRate but applied the
      // regularization term unscaled, so at lr = 0.1 the penalty landed ten
      // times harder than configured. Both terms are now combined first and
      // the learning rate applied once to the sum, which is the standard
      // update W := W - lr * (dW + reg).
      let step = gradW;
      if (this.regType === 'l2' && this.regularization > 0) {
        step = step.add(this.weights[i].scale(this.regularization / m));
      } else if (this.regType === 'l1' && this.regularization > 0) {
        step = step.add(this.weights[i].apply((w) => (this.regularization / m) * Math.sign(w)));
      }
      this.weights[i] = this.weights[i].subtract(step.scale(this.learningRate));

      const dB = new Matrix(1, deltas[i].cols);
      for (let j = 0; j < deltas[i].cols; j++) {
        let sum = 0;
        for (let k = 0; k < deltas[i].rows; k++) sum += deltas[i].data[k][j];
        dB.data[0][j] = (sum / m) * this.learningRate;
      }
      this.biases[i] = this.biases[i].subtract(dB);
    }
  }

  /** @param {Matrix} input @returns {Matrix} */
  predict(input) {
    return this.forward(input);
  }

  /** @returns {number[]} neuron counts including the input layer */
  getArchitecture() {
    return [this.inputSize, ...this.layers];
  }
}
