/**
 * Deterministic pseudo-random number generation.
 *
 * The original demo held a single module-level `rng` binding that every
 * consumer reached for implicitly. That made dataset generation, weight
 * initialisation and mini-batch sampling impossible to test in isolation and
 * impossible to reason about when two of them interleaved. Here the generator
 * is a value you create and pass explicitly.
 */

/**
 * mulberry32 — a fast 32-bit seeded PRNG.
 * @param {number} seed
 * @returns {() => number} generator yielding floats in [0, 1)
 */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Create a seeded generator. Alias of {@link mulberry32} with a clearer name
 * at call sites.
 * @param {number} seed
 * @returns {() => number}
 */
export function makeRng(seed) {
  return mulberry32(seed);
}
