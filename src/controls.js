/** Input sanitising for the playground's form controls. Pure, so it is unit-tested. */

export const MIN_NEURONS = 1;
export const MAX_NEURONS = 32;

/**
 * A hidden layer's neuron count from whatever the number box holds: rounded to a whole
 * number and clamped to 1–32. A fractional count reached `new Array(n)` and threw
 * "RangeError: Invalid array length".
 * @param {unknown} value
 * @returns {number}
 */
export function clampNeurons(value) {
  const n = Math.round(Number(value));
  if (!Number.isFinite(n)) return MIN_NEURONS;
  return Math.max(MIN_NEURONS, Math.min(MAX_NEURONS, n));
}
