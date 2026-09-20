import { describe, it, expect } from 'vitest';
import { mulberry32, makeRng } from '../src/rng.js';

describe('mulberry32', () => {
  it('is deterministic for a given seed', () => {
    const a = mulberry32(42);
    const b = mulberry32(42);
    const seqA = Array.from({ length: 10 }, () => a());
    const seqB = Array.from({ length: 10 }, () => b());
    expect(seqA).toEqual(seqB);
  });

  it('produces different sequences for different seeds', () => {
    const a = mulberry32(1);
    const b = mulberry32(2);
    expect(a()).not.toBe(b());
  });

  it('yields values in [0, 1)', () => {
    const r = mulberry32(7);
    for (let i = 0; i < 500; i++) {
      const v = r();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  it('handles seed 0 without collapsing to a constant', () => {
    const r = mulberry32(0);
    const vals = new Set(Array.from({ length: 20 }, () => r()));
    expect(vals.size).toBeGreaterThan(1);
  });

  it('makeRng is an alias with the same behaviour', () => {
    expect(Array.from({ length: 5 }, makeRng(3))).toEqual(
      Array.from({ length: 5 }, mulberry32(3))
    );
  });
});
