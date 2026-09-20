import { makeRng } from './rng.js';

/** The four synthetic classification problems the playground ships with. */
export const DATASET_TYPES = ['xor', 'circle', 'spiral', 'gaussian', 'moons', 'sine'];

/**
 * Generate a labelled 2-D dataset in the unit square.
 *
 * @param {string} type one of {@link DATASET_TYPES}
 * @param {number} numPoints
 * @param {{rng?: () => number, seed?: number}} [options]
 *        Pass `rng` to supply a generator, or `seed` to build one. Defaults to
 *        seed 42 so a call with no options is still reproducible.
 * @returns {{data: number[][], labels: number[]}}
 */
export function generateDataset(type, numPoints = 300, options = {}) {
  if (!DATASET_TYPES.includes(type)) {
    throw new Error(`Unknown dataset type: ${type}`);
  }
  if (!Number.isInteger(numPoints) || numPoints < 1) {
    throw new Error('numPoints must be a positive integer');
  }
  const rng = options.rng ?? makeRng(options.seed ?? 42);

  const data = [];
  const labels = [];

  if (type === 'xor') {
    const corners = [
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1]
    ];
    const expected = [0, 1, 1, 0];
    for (let i = 0; i < numPoints; i++) {
      const corner = corners[i % 4];
      const noise = 0.1;
      data.push([corner[0] + (rng() - 0.5) * noise, corner[1] + (rng() - 0.5) * noise]);
      labels.push(expected[i % 4]);
    }
  } else if (type === 'circle') {
    for (let i = 0; i < numPoints; i++) {
      const angle = rng() * 2 * Math.PI;
      // sqrt keeps the sample uniform over the disc's *area*; the original used
      // the raw uniform, which crowded points toward the centre.
      const unit = Math.sqrt(rng());
      const x = 0.5 + unit * 0.4 * Math.cos(angle);
      const y = 0.5 + unit * 0.4 * Math.sin(angle);
      data.push([x, y]);
      labels.push(unit < Math.SQRT1_2 ? 0 : 1);
    }
  } else if (type === 'spiral') {
    // Two *interleaved* arms. The original gave both classes the same angle and
    // radius progression, so class 0 was simply the inner half of one spiral and
    // class 1 the outer half — separable by radius alone, which defeats the
    // point of the spiral benchmark. Each class now gets its own arm, offset by
    // pi, and its own sweep from the centre outward.
    const half = Math.ceil(numPoints / 2);
    for (let i = 0; i < numPoints; i++) {
      const label = i < half ? 0 : 1;
      const withinClass = label === 0 ? i : i - half;
      const span = label === 0 ? half : numPoints - half;
      const t = span <= 1 ? 0 : withinClass / (span - 1);
      const angle = t * 3 * Math.PI + label * Math.PI;
      const radius = 0.05 + t * 0.35;
      const noise = (rng() - 0.5) * 0.05;
      data.push([
        0.5 + (radius + noise) * Math.cos(angle),
        0.5 + (radius + noise) * Math.sin(angle)
      ]);
      labels.push(label);
    }
  } else if (type === 'moons') {
    // Two interleaving half-circles. The UI has always offered this button, but
    // the original generator had no branch for it and fell through to gaussian,
    // so "Moons" silently produced two blobs.
    const half = Math.ceil(numPoints / 2);
    for (let i = 0; i < numPoints; i++) {
      const label = i < half ? 0 : 1;
      const withinClass = label === 0 ? i : i - half;
      const span = label === 0 ? half : numPoints - half;
      const t = span <= 1 ? 0 : withinClass / (span - 1);
      const angle = t * Math.PI;
      const jitter = () => (rng() - 0.5) * 0.06;
      if (label === 0) {
        data.push([0.35 + 0.3 * Math.cos(angle) + jitter(), 0.58 + 0.3 * Math.sin(angle) + jitter()]);
      } else {
        data.push([0.65 - 0.3 * Math.cos(angle) + jitter(), 0.42 - 0.3 * Math.sin(angle) + jitter()]);
      }
      labels.push(label);
    }
  } else if (type === 'sine') {
    // Label is the side of a sine wave a point falls on — a smooth, genuinely
    // non-linear boundary. Also absent from the original generator.
    for (let i = 0; i < numPoints; i++) {
      const x = rng();
      const y = rng();
      const boundary = 0.5 + 0.28 * Math.sin(x * 2 * Math.PI);
      data.push([x, y]);
      labels.push(y < boundary ? 0 : 1);
    }
  } else {
    // gaussian
    for (let i = 0; i < numPoints; i++) {
      const label = i < numPoints / 2 ? 0 : 1;
      const centerX = label === 0 ? 0.3 : 0.7;
      const centerY = label === 0 ? 0.3 : 0.7;
      data.push([centerX + (rng() - 0.5) * 0.3, centerY + (rng() - 0.5) * 0.3]);
      labels.push(label);
    }
  }

  return { data, labels };
}

/**
 * One-hot encode binary labels into the 2-column target matrix the network
 * expects.
 * @param {number[]} labels
 * @returns {number[][]}
 */
export function oneHot(labels) {
  return labels.map((l) => (l === 0 ? [1, 0] : [0, 1]));
}
