import { describe, it, expect } from 'vitest';
import {
  relu, reluDerivative, sigmoid, sigmoidDerivative,
  tanh, tanhDerivative, linear, linearDerivative,
  softmax, getActivation
} from '../src/activation.js';

describe('activations', () => {
  it('relu clamps negatives to zero', () => {
    expect(relu(-3)).toBe(0);
    expect(relu(2.5)).toBe(2.5);
    expect(reluDerivative(-1)).toBe(0);
    expect(reluDerivative(1)).toBe(1);
  });

  it('sigmoid is centred at 0.5 and monotonic', () => {
    expect(sigmoid(0)).toBeCloseTo(0.5, 10);
    expect(sigmoid(10)).toBeGreaterThan(sigmoid(1));
    expect(sigmoidDerivative(0.5)).toBeCloseTo(0.25, 10);
  });

  it('tanh is odd and bounded', () => {
    expect(tanh(0)).toBe(0);
    expect(tanh(-1)).toBeCloseTo(-tanh(1), 10);
    expect(tanhDerivative(0)).toBe(1);
  });

  it('linear is the identity with unit slope', () => {
    expect(linear(7)).toBe(7);
    expect(linearDerivative()).toBe(1);
  });
});

describe('softmax', () => {
  it('sums to one', () => {
    const out = softmax([1, 2, 3]);
    expect(out.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
  });

  it('is monotonic in the input', () => {
    const out = softmax([1, 2, 3]);
    expect(out[2]).toBeGreaterThan(out[1]);
    expect(out[1]).toBeGreaterThan(out[0]);
  });

  it('is numerically stable for large logits', () => {
    // Without the max-subtraction this overflows to NaN.
    const out = softmax([1000, 1001]);
    expect(out.every(Number.isFinite)).toBe(true);
    expect(out.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
  });

  it('is uniform for equal logits', () => {
    expect(softmax([5, 5, 5])).toEqual([1 / 3, 1 / 3, 1 / 3].map((v) => expect.closeTo(v, 10)));
  });
});

describe('getActivation', () => {
  it('resolves known names', () => {
    expect(getActivation('relu').fn(-1)).toBe(0);
  });
  it('throws on unknown names', () => {
    expect(() => getActivation('nope')).toThrow(/Unknown activation/);
  });
});
