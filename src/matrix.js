/**
 * A minimal dense matrix. Every operation returns a new Matrix; nothing
 * mutates in place except `randomize`, which is explicit about it.
 */
export class Matrix {
  /**
   * @param {number} rows
   * @param {number} cols
   */
  constructor(rows, cols) {
    this.rows = rows;
    this.cols = cols;
    this.data = Array.from({ length: rows }, () => new Array(cols).fill(0));
  }

  /**
   * Build a Matrix from a plain 2-D array.
   * @param {number[][]} values
   * @returns {Matrix}
   */
  static from(values) {
    const m = new Matrix(values.length, values[0]?.length ?? 0);
    for (let i = 0; i < m.rows; i++) {
      for (let j = 0; j < m.cols; j++) m.data[i][j] = values[i][j];
    }
    return m;
  }

  /**
   * Fill with uniform noise in [-1, 1).
   * The generator is injected so initialisation is reproducible and testable.
   * @param {() => number} rng
   * @returns {Matrix} this
   */
  randomize(rng) {
    if (typeof rng !== 'function') {
      throw new TypeError('Matrix.randomize requires an rng function');
    }
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) this.data[i][j] = rng() * 2 - 1;
    }
    return this;
  }

  /** @param {Matrix} other @returns {Matrix} */
  multiply(other) {
    if (this.cols !== other.rows) {
      throw new Error(
        `Incompatible dimensions: ${this.rows}x${this.cols} * ${other.rows}x${other.cols}`
      );
    }
    const result = new Matrix(this.rows, other.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < other.cols; j++) {
        let sum = 0;
        for (let k = 0; k < this.cols; k++) sum += this.data[i][k] * other.data[k][j];
        result.data[i][j] = sum;
      }
    }
    return result;
  }

  /**
   * Elementwise add, with row broadcasting for bias vectors.
   *
   * A `1 x cols` operand is added to every row. The original implementation
   * required an exact shape match, which meant `activations.add(bias)` threw
   * `Incompatible dimensions` for any batch larger than one row — and the UI
   * ships a default batch size of 32, so training threw on the first forward
   * pass. Broadcasting is what a bias add means, and it is what the network
   * always needed.
   *
   * @param {Matrix} other
   * @returns {Matrix}
   */
  add(other) {
    if (other.rows === 1 && this.rows !== 1 && other.cols === this.cols) {
      const result = new Matrix(this.rows, this.cols);
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
          result.data[i][j] = this.data[i][j] + other.data[0][j];
        }
      }
      return result;
    }
    this.#assertSameShape(other, 'add');
    return this.#zip(other, (a, b) => a + b);
  }

  /** @param {Matrix} other @returns {Matrix} */
  subtract(other) {
    this.#assertSameShape(other, 'subtract');
    return this.#zip(other, (a, b) => a - b);
  }

  /** @param {number} scalar @returns {Matrix} */
  scale(scalar) {
    return this.apply((v) => v * scalar);
  }

  /** @returns {Matrix} */
  transpose() {
    const result = new Matrix(this.cols, this.rows);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) result.data[j][i] = this.data[i][j];
    }
    return result;
  }

  /** @param {(v: number) => number} fn @returns {Matrix} */
  apply(fn) {
    const result = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) result.data[i][j] = fn(this.data[i][j]);
    }
    return result;
  }

  /** @returns {Matrix} */
  copy() {
    return this.apply((v) => v);
  }

  /**
   * The original `subtract` skipped the dimension check that `add` performed,
   * so a shape mismatch produced silent NaNs instead of an error. Both now go
   * through this guard.
   */
  #assertSameShape(other, op) {
    if (this.rows !== other.rows || this.cols !== other.cols) {
      throw new Error(
        `Incompatible dimensions for ${op}: ${this.rows}x${this.cols} vs ${other.rows}x${other.cols}`
      );
    }
  }

  #zip(other, fn) {
    const result = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        result.data[i][j] = fn(this.data[i][j], other.data[i][j]);
      }
    }
    return result;
  }
}
