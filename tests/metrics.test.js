import { describe, it, expect } from 'vitest';
import { calculateLoss, calculateAccuracy, sampleBatchIndices } from '../src/metrics.js';
import { Matrix } from '../src/matrix.js';
import { mulberry32 } from '../src/rng.js';

describe('calculateLoss', () => {
  it('is near zero for confident correct predictions', () => {
    const pred = Matrix.from([[0.999, 0.001]]);
    const target = Matrix.from([[1, 0]]);
    expect(calculateLoss(pred, target)).toBeLessThan(0.01);
  });

  it('is large for confident wrong predictions', () => {
    const pred = Matrix.from([[0.001, 0.999]]);
    const target = Matrix.from([[1, 0]]);
    expect(calculateLoss(pred, target)).toBeGreaterThan(5);
  });

  it('stays finite when a predicted probability is exactly zero', () => {
    const pred = Matrix.from([[0, 1]]);
    const target = Matrix.from([[1, 0]]);
    expect(Number.isFinite(calculateLoss(pred, target))).toBe(true);
  });

  it('averages across rows', () => {
    const pred = Matrix.from([
      [0.5, 0.5],
      [0.5, 0.5]
    ]);
    const target = Matrix.from([
      [1, 0],
      [1, 0]
    ]);
    expect(calculateLoss(pred, target)).toBeCloseTo(Math.log(2), 6);
  });
});

describe('calculateAccuracy', () => {
  it('scores a perfect classifier at 1', () => {
    const pred = Matrix.from([
      [0.9, 0.1],
      [0.2, 0.8]
    ]);
    expect(calculateAccuracy(pred, [0, 1])).toBe(1);
  });

  it('scores an inverted classifier at 0', () => {
    const pred = Matrix.from([
      [0.9, 0.1],
      [0.2, 0.8]
    ]);
    expect(calculateAccuracy(pred, [1, 0])).toBe(0);
  });

  it('handles ties by preferring the first column', () => {
    const pred = Matrix.from([[0.5, 0.5]]);
    expect(calculateAccuracy(pred, [0])).toBe(1);
  });

  it('returns 0 for an empty prediction set', () => {
    expect(calculateAccuracy(new Matrix(0, 2), [])).toBe(0);
  });
});

describe('sampleBatchIndices', () => {
  it('returns batchSize distinct indices', () => {
    const idx = sampleBatchIndices(100, 16, mulberry32(1));
    expect(idx).toHaveLength(16);
    expect(new Set(idx).size).toBe(16);
  });

  it('clamps to the dataset size', () => {
    expect(sampleBatchIndices(5, 50, mulberry32(1))).toHaveLength(5);
  });

  it('is reproducible for a seed', () => {
    expect(sampleBatchIndices(50, 10, mulberry32(8))).toEqual(
      sampleBatchIndices(50, 10, mulberry32(8))
    );
  });

  it('only returns in-range indices', () => {
    for (const i of sampleBatchIndices(30, 30, mulberry32(2))) {
      expect(i).toBeGreaterThanOrEqual(0);
      expect(i).toBeLessThan(30);
    }
  });
});
