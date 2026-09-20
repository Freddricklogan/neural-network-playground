/**
 * Canvas rendering. Everything here touches the DOM and nothing here computes
 * anything the pure layer already knows how to compute.
 */

const CLASS_COLOURS = ['#58A6FF', '#f85149'];

/**
 * Chart.js is loaded from a CDN with SRI. If it is blocked — offline, a locked
 * -down network, a CI smoke run — `window.Chart` is undefined. Rather than
 * throwing (which in the original would have aborted every later render), we
 * fall back to the vendored copy, and failing that degrade gracefully.
 * @returns {Promise<unknown|null>}
 */
export async function loadChartLib() {
  if (globalThis.Chart) return globalThis.Chart;
  try {
    await import('../vendor/chart.umd.min.js');
  } catch {
    return null;
  }
  return globalThis.Chart ?? null;
}

/**
 * @param {HTMLCanvasElement} canvas
 * @param {unknown} ChartLib
 * @returns {{push(epoch:number, loss:number, acc:number):void, reset():void}|null}
 */
export function createLossChart(canvas, ChartLib) {
  if (!ChartLib || !canvas) return null;
  const chart = new ChartLib(canvas.getContext('2d'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'Loss', data: [], borderColor: '#f85149', backgroundColor: 'rgba(248,81,73,.12)', tension: 0.25, pointRadius: 0, yAxisID: 'y' },
        { label: 'Accuracy', data: [], borderColor: '#3fb950', backgroundColor: 'rgba(63,185,80,.12)', tension: 0.25, pointRadius: 0, yAxisID: 'y1' }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { ticks: { color: '#8b98b0' }, grid: { color: 'rgba(34,48,77,.6)' } },
        y: { position: 'left', ticks: { color: '#8b98b0' }, grid: { color: 'rgba(34,48,77,.6)' }, title: { display: true, text: 'Loss', color: '#8b98b0' } },
        y1: { position: 'right', min: 0, max: 1, ticks: { color: '#8b98b0' }, grid: { drawOnChartArea: false }, title: { display: true, text: 'Accuracy', color: '#8b98b0' } }
      },
      plugins: { legend: { labels: { color: '#e6edf3' } } }
    }
  });

  return {
    push(epoch, loss, acc) {
      chart.data.labels.push(epoch);
      chart.data.datasets[0].data.push(loss);
      chart.data.datasets[1].data.push(acc);
      chart.update('none');
    },
    reset() {
      chart.data.labels = [];
      chart.data.datasets[0].data = [];
      chart.data.datasets[1].data = [];
      chart.update('none');
    }
  };
}

/**
 * Paint the network's decision surface, then the training points on top.
 * @param {HTMLCanvasElement} canvas
 * @param {import('./network.js').NeuralNetwork} network
 * @param {number[][]} data
 * @param {number[]} labels
 * @param {(rows:number[][]) => import('./matrix.js').Matrix} toMatrix
 */
export function drawDecisionBoundary(canvas, network, data, labels, toMatrix) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const { width, height } = canvas;
  const step = 6;

  const points = [];
  for (let py = 0; py < height; py += step) {
    for (let px = 0; px < width; px += step) points.push([px / width, 1 - py / height]);
  }
  const preds = network.predict(toMatrix(points));

  let k = 0;
  for (let py = 0; py < height; py += step) {
    for (let px = 0; px < width; px += step) {
      const p1 = preds.data[k++][1];
      const mix = Math.max(0, Math.min(1, p1));
      ctx.fillStyle = `rgba(${Math.round(88 + (248 - 88) * mix)}, ${Math.round(166 + (81 - 166) * mix)}, ${Math.round(255 + (73 - 255) * mix)}, 0.30)`;
      ctx.fillRect(px, py, step, step);
    }
  }

  for (let i = 0; i < data.length; i++) {
    const [x, y] = data[i];
    ctx.beginPath();
    ctx.arc(x * width, (1 - y) * height, 3.2, 0, Math.PI * 2);
    ctx.fillStyle = CLASS_COLOURS[labels[i]] ?? CLASS_COLOURS[0];
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = 'rgba(11,18,32,.85)';
    ctx.stroke();
  }
}

/**
 * Draw the layer/neuron diagram.
 * @param {HTMLCanvasElement} canvas
 * @param {number[]} architecture neuron counts, input layer first
 */
export function drawNetworkDiagram(canvas, architecture) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);

  const padX = 34;
  const usable = width - padX * 2;
  const gapX = architecture.length > 1 ? usable / (architecture.length - 1) : 0;

  const positions = architecture.map((count, li) => {
    const x = padX + gapX * li;
    const gapY = height / (count + 1);
    return Array.from({ length: count }, (_, ni) => ({ x, y: gapY * (ni + 1) }));
  });

  ctx.strokeStyle = 'rgba(88,166,255,.22)';
  ctx.lineWidth = 1;
  for (let li = 0; li < positions.length - 1; li++) {
    for (const a of positions[li]) {
      for (const b of positions[li + 1]) {
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
    }
  }

  positions.forEach((layer, li) => {
    const colour = li === 0 ? '#3fb950' : li === positions.length - 1 ? '#d2a8ff' : '#58A6FF';
    for (const node of layer) {
      ctx.beginPath();
      ctx.arc(node.x, node.y, 6, 0, Math.PI * 2);
      ctx.fillStyle = colour;
      ctx.fill();
    }
  });
}
