/**
 * Training metrics. All pure: they take matrices and arrays and return numbers.
 */

/**
 * Mean categorical cross-entropy.
 * Predictions are clamped away from zero before the log so a confident-but-wrong
 * prediction yields a large finite loss rather than Infinity.
 *
 * @param {import('./matrix.js').Matrix} predictions
 * @param {import('./matrix.js').Matrix} target one-hot
 * @returns {number}
 */
export function calculateLoss(predictions, target) {
  let loss = 0;
  for (let i = 0; i < predictions.rows; i++) {
    for (let c = 0; c < predictions.cols; c++) {
      loss += -(target.data[i][c] * Math.log(Math.max(predictions.data[i][c], 1e-10)));
    }
  }
  return loss / predictions.rows;
}

/**
 * Fraction of rows whose arg-max column equals the integer label.
 * @param {import('./matrix.js').Matrix} predictions
 * @param {number[]} labels
 * @returns {number} in [0, 1]
 */
export function calculateAccuracy(predictions, labels) {
  let correct = 0;
  for (let i = 0; i < predictions.rows; i++) {
    let best = 0;
    for (let c = 1; c < predictions.cols; c++) {
      if (predictions.data[i][c] > predictions.data[i][best]) best = c;
    }
    if (best === labels[i]) correct++;
  }
  return predictions.rows === 0 ? 0 : correct / predictions.rows;
}

/**
 * Deterministic mini-batch sampling without replacement.
 * @param {number} total
 * @param {number} batchSize
 * @param {() => number} rng
 * @returns {number[]} indices
 */
export function sampleBatchIndices(total, batchSize, rng) {
  const size = Math.min(batchSize, total);
  const pool = Array.from({ length: total }, (_, i) => i);
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  return pool.slice(0, size);
}
