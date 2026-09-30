import { describe, expect, it } from 'vitest';
import { clampNeurons, MAX_NEURONS, MIN_NEURONS } from '../src/controls.js';
import { Matrix } from '../src/matrix.js';

describe('clampNeurons', () => {
  it('keeps a whole number inside the range', () => {
    expect(clampNeurons('12')).toBe(12);
  });
  it('rounds fractions, which used to crash the network build', () => {
    expect(clampNeurons('1.5')).toBe(2);
    expect(clampNeurons('16.25')).toBe(16);
    expect(() => new Matrix(2, clampNeurons('16.25'))).not.toThrow();
  });
  it('clamps below and above the range', () => {
    expect(clampNeurons('-3')).toBe(MIN_NEURONS);
    expect(clampNeurons('0')).toBe(MIN_NEURONS);
    expect(clampNeurons('999')).toBe(MAX_NEURONS);
  });
  it('falls back to the minimum for empty or non-numeric input', () => {
    expect(clampNeurons('')).toBe(MIN_NEURONS);
    expect(clampNeurons('abc')).toBe(MIN_NEURONS);
    expect(clampNeurons(undefined)).toBe(MIN_NEURONS);
    expect(clampNeurons(Infinity)).toBe(MIN_NEURONS);
  });
});
