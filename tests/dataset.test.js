import { describe, it, expect } from 'vitest';
import { generateDataset, oneHot, DATASET_TYPES } from '../src/dataset.js';
import { mulberry32 } from '../src/rng.js';

describe('generateDataset', () => {
  it.each(DATASET_TYPES)('%s returns the requested number of points', (type) => {
    const { data, labels } = generateDataset(type, 120, { seed: 1 });
    expect(data).toHaveLength(120);
    expect(labels).toHaveLength(120);
  });

  it.each(DATASET_TYPES)('%s produces both classes', (type) => {
    const { labels } = generateDataset(type, 200, { seed: 3 });
    expect(new Set(labels)).toEqual(new Set([0, 1]));
  });

  it('is reproducible for a given seed', () => {
    const a = generateDataset('spiral', 50, { seed: 9 });
    const b = generateDataset('spiral', 50, { seed: 9 });
    expect(a.data).toEqual(b.data);
  });

  it('accepts an injected rng', () => {
    const a = generateDataset('xor', 20, { rng: mulberry32(4) });
    const b = generateDataset('xor', 20, { rng: mulberry32(4) });
    expect(a.data).toEqual(b.data);
  });

  it('rejects unknown types and bad counts', () => {
    expect(() => generateDataset('nope', 10)).toThrow(/Unknown dataset type/);
    expect(() => generateDataset('xor', 0)).toThrow(/positive integer/);
  });

  it('xor labels follow the exclusive-or of the corner', () => {
    const { data, labels } = generateDataset('xor', 4, { seed: 1 });
    data.forEach(([x, y], i) => {
      expect(labels[i]).toBe((Math.round(x) ^ Math.round(y)) === 1 ? 1 : 0);
    });
  });

  it('spiral arms are NOT separable by radius alone', () => {
    // Regression test for the original generator, which gave both classes the
    // same angle/radius sweep so class 0 was the inner half of one spiral and
    // class 1 the outer half — a radial threshold classified it perfectly.
    const { data, labels } = generateDataset('spiral', 400, { seed: 11 });
    const radius = ([x, y]) => Math.hypot(x - 0.5, y - 0.5);
    const r0 = data.filter((_, i) => labels[i] === 0).map(radius);
    const r1 = data.filter((_, i) => labels[i] === 1).map(radius);
    // The two classes must overlap in radius; disjoint ranges mean the bug.
    expect(Math.min(...r0)).toBeLessThan(Math.max(...r1));
    expect(Math.min(...r1)).toBeLessThan(Math.max(...r0));

    // And no single radial threshold should separate them well.
    let bestAccuracy = 0;
    for (let t = 0; t <= 0.5; t += 0.005) {
      const correct = data.filter((p, i) => (radius(p) < t ? 0 : 1) === labels[i]).length;
      bestAccuracy = Math.max(bestAccuracy, correct / data.length);
    }
    expect(bestAccuracy).toBeLessThan(0.8);
  });

  it('circle samples are spread over the disc, not crowded at the centre', () => {
    const { data } = generateDataset('circle', 1000, { seed: 2 });
    const radii = data.map(([x, y]) => Math.hypot(x - 0.5, y - 0.5));
    const outerHalf = radii.filter((r) => r > 0.2).length;
    // Uniform-by-area puts ~75% of points beyond half the max radius.
    expect(outerHalf / radii.length).toBeGreaterThan(0.6);
  });

  it('all points stay within the unit square for every type', () => {
    for (const type of DATASET_TYPES) {
      const { data } = generateDataset(type, 300, { seed: 6 });
      for (const [x, y] of data) {
        expect(x).toBeGreaterThan(-0.3);
        expect(x).toBeLessThan(1.3);
        expect(y).toBeGreaterThan(-0.3);
        expect(y).toBeLessThan(1.3);
      }
    }
  });
});

describe('moons and sine (previously unimplemented)', () => {
  it('moons is not two gaussian blobs', () => {
    // The original had no 'moons' branch, so the button fell through to
    // gaussian. Gaussian blobs are linearly separable; interleaved moons are not.
    const { data, labels } = generateDataset('moons', 400, { seed: 4 });
    let best = 0;
    for (let t = 0; t <= 1; t += 0.01) {
      const correct = data.filter(([x], i) => (x < t ? 0 : 1) === labels[i]).length;
      best = Math.max(best, correct / data.length, 1 - correct / data.length);
    }
    expect(best).toBeLessThan(0.95);
  });

  it('moons produces two arcs that overlap horizontally', () => {
    const { data, labels } = generateDataset('moons', 300, { seed: 5 });
    const xs0 = data.filter((_, i) => labels[i] === 0).map(([x]) => x);
    const xs1 = data.filter((_, i) => labels[i] === 1).map(([x]) => x);
    expect(Math.max(...xs0)).toBeGreaterThan(Math.min(...xs1));
  });

  it('sine labels match the side of the wave', () => {
    const { data, labels } = generateDataset('sine', 200, { seed: 6 });
    data.forEach(([x, y], i) => {
      const boundary = 0.5 + 0.28 * Math.sin(x * 2 * Math.PI);
      expect(labels[i]).toBe(y < boundary ? 0 : 1);
    });
  });

  it('sine is not separable by a horizontal line', () => {
    const { data, labels } = generateDataset('sine', 400, { seed: 7 });
    let best = 0;
    for (let t = 0; t <= 1; t += 0.01) {
      const correct = data.filter(([, y], i) => (y < t ? 0 : 1) === labels[i]).length;
      best = Math.max(best, correct / data.length);
    }
    expect(best).toBeLessThan(0.9);
  });
});

describe('oneHot', () => {
  it('maps 0 and 1 to two columns', () => {
    expect(oneHot([0, 1, 0])).toEqual([
      [1, 0],
      [0, 1],
      [1, 0]
    ]);
  });
});
