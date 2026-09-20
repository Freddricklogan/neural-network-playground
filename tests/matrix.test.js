import { describe, it, expect } from 'vitest';
import { Matrix } from '../src/matrix.js';
import { mulberry32 } from '../src/rng.js';

describe('Matrix', () => {
  it('constructs zero-filled with the given shape', () => {
    const m = new Matrix(2, 3);
    expect(m.rows).toBe(2);
    expect(m.cols).toBe(3);
    expect(m.data).toEqual([
      [0, 0, 0],
      [0, 0, 0]
    ]);
  });

  it('builds from a plain array', () => {
    const m = Matrix.from([
      [1, 2],
      [3, 4]
    ]);
    expect(m.rows).toBe(2);
    expect(m.data[1][0]).toBe(3);
  });

  it('multiplies conformable matrices', () => {
    const a = Matrix.from([
      [1, 2],
      [3, 4]
    ]);
    const b = Matrix.from([
      [5, 6],
      [7, 8]
    ]);
    expect(a.multiply(b).data).toEqual([
      [19, 22],
      [43, 50]
    ]);
  });

  it('throws on non-conformable multiply', () => {
    expect(() => new Matrix(2, 3).multiply(new Matrix(2, 3))).toThrow(/Incompatible/);
  });

  it('adds and subtracts elementwise', () => {
    const a = Matrix.from([[1, 2]]);
    const b = Matrix.from([[10, 20]]);
    expect(a.add(b).data).toEqual([[11, 22]]);
    expect(b.subtract(a).data).toEqual([[9, 18]]);
  });

  it('broadcasts a 1-row bias across every row', () => {
    // Regression test: the original threw "Incompatible dimensions" here, which
    // made training fail on the first forward pass for any batch size > 1.
    const activations = Matrix.from([
      [1, 2],
      [3, 4],
      [5, 6]
    ]);
    const bias = Matrix.from([[10, 20]]);
    expect(activations.add(bias).data).toEqual([
      [11, 22],
      [13, 24],
      [15, 26]
    ]);
  });

  it('still rejects genuinely mismatched shapes in add', () => {
    expect(() => new Matrix(3, 2).add(new Matrix(2, 3))).toThrow(/Incompatible/);
  });

  it('throws when subtract shapes disagree', () => {
    // The original skipped this guard and produced silent NaNs.
    expect(() => new Matrix(1, 2).subtract(new Matrix(2, 2))).toThrow(/Incompatible/);
  });

  it('scales and transposes', () => {
    const m = Matrix.from([
      [1, 2],
      [3, 4]
    ]);
    expect(m.scale(2).data).toEqual([
      [2, 4],
      [6, 8]
    ]);
    expect(m.transpose().data).toEqual([
      [1, 3],
      [2, 4]
    ]);
  });

  it('applies a function elementwise without mutating the source', () => {
    const m = Matrix.from([[1, 2]]);
    const doubled = m.apply((v) => v * 2);
    expect(doubled.data).toEqual([[2, 4]]);
    expect(m.data).toEqual([[1, 2]]);
  });

  it('copies deeply', () => {
    const m = Matrix.from([[1]]);
    const c = m.copy();
    c.data[0][0] = 99;
    expect(m.data[0][0]).toBe(1);
  });

  it('randomize requires an injected rng', () => {
    expect(() => new Matrix(1, 1).randomize()).toThrow(TypeError);
  });

  it('randomize is reproducible for a seed and stays in [-1, 1)', () => {
    const a = new Matrix(3, 3).randomize(mulberry32(5));
    const b = new Matrix(3, 3).randomize(mulberry32(5));
    expect(a.data).toEqual(b.data);
    for (const row of a.data) {
      for (const v of row) {
        expect(v).toBeGreaterThanOrEqual(-1);
        expect(v).toBeLessThan(1);
      }
    }
  });
});
