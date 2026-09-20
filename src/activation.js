/**
 * Activation functions and their derivatives.
 *
 * Note on the `*Derivative` signatures: sigmoid, tanh and softmax derivatives
 * take the *activated* value (the layer output), not the pre-activation. ReLU
 * and linear take either, since their derivative depends only on the sign.
 * This matches how the backward pass calls them.
 */

/** @param {number} x */
export const relu = (x) => Math.max(0, x);
/** @param {number} x */
export const reluDerivative = (x) => (x > 0 ? 1 : 0);

/** @param {number} x */
export const sigmoid = (x) => 1 / (1 + Math.exp(-x));
/** @param {number} a activated value */
export const sigmoidDerivative = (a) => a * (1 - a);

/** @param {number} x */
export const tanh = (x) => Math.tanh(x);
/** @param {number} a activated value */
export const tanhDerivative = (a) => 1 - a * a;

/** @param {number} x */
export const linear = (x) => x;
/** @returns {number} */
export const linearDerivative = () => 1;

/**
 * Numerically stable softmax over a vector.
 * Subtracting the max before exponentiating prevents overflow for large logits.
 * @param {number[]} vec
 * @returns {number[]}
 */
export function softmax(vec) {
  const max = Math.max(...vec);
  const exps = vec.map((v) => Math.exp(v - max));
  const sum = exps.reduce((acc, v) => acc + v, 0);
  return exps.map((v) => v / sum);
}

/** @param {number} a activated value */
export const softmaxDerivative = (a) => a * (1 - a);

/** Lookup table used by the network to resolve a name to a pair. */
export const ACTIVATIONS = {
  relu: { fn: relu, dfn: reluDerivative },
  sigmoid: { fn: sigmoid, dfn: sigmoidDerivative },
  tanh: { fn: tanh, dfn: tanhDerivative },
  linear: { fn: linear, dfn: linearDerivative }
};

/**
 * @param {string} name
 * @returns {{fn: (x:number)=>number, dfn: (x:number)=>number}}
 */
export function getActivation(name) {
  const found = ACTIVATIONS[name];
  if (!found) throw new Error(`Unknown activation: ${name}`);
  return found;
}
