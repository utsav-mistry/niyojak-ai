/**
 * admin.js — Admin portal client
 *
 * Connects to the /admin/events SSE stream and updates the DOM in real time:
 *   - Node status cards (AI score, CPU %, Memory %)
 *   - Pod placement map (which pod is on which node)
 *   - Flood statistics (RPS, success, failed)
 *
 * Also handles button clicks for stress/release and flood start/stop.
 */

"use strict";

// ---------------------------------------------------------------------------
// SSE connection
// ---------------------------------------------------------------------------

let evtSource = null;
let lastPodNames = new Set();

function connectSSE() {
  if (evtSource) evtSource.close();

  evtSource = new EventSource('/admin/events');

  evtSource.onopen = () => {
    document.getElementById('conn-status').className = 'conn-badge';
    document.getElementById('conn-status').innerHTML =
      '<span class="dot dot--green dot--pulse"></span> Live';
  };

  evtSource.onmessage = (e) => {
    const status = JSON.parse(e.data);
    updateAll(status);
  };

  evtSource.onerror = () => {
    document.getElementById('conn-status').className = 'conn-badge conn-badge--disconnected';
    document.getElementById('conn-status').innerHTML =
      '<span class="dot dot--red"></span> Disconnected';
    document.getElementById('last-update').textContent = 'Reconnecting...';
    // Auto-reconnect after 3s
    setTimeout(connectSSE, 3000);
  };
}

// ---------------------------------------------------------------------------
// Main update dispatcher
// ---------------------------------------------------------------------------

function updateAll(status) {
  document.getElementById('last-update').textContent =
    'Updated ' + new Date(status.ts).toLocaleTimeString();
  document.getElementById('todo-count').textContent =
    'Tasks: ' + status.todoCount;

  renderNodes(status.nodes, status.stressedNodes || []);
  renderPodMap(status.pods || []);
  renderFloodStats(status.flood);
}

// ---------------------------------------------------------------------------
// Node cards
// ---------------------------------------------------------------------------

function renderNodes(nodes, stressedNodes) {
  const grid = document.getElementById('node-grid');

  if (!nodes || nodes.length === 0) {
    grid.innerHTML = '<div class="card" style="opacity:0.5;text-align:center;padding:2rem"><p class="text-muted text-sm">AI service not reachable — node scores unavailable</p></div>';
    return;
  }

  grid.innerHTML = nodes.map(node => nodeCardHTML(node, stressedNodes)).join('');

  // Attach stress/release button handlers
  grid.querySelectorAll('[data-stress]').forEach(btn => {
    btn.addEventListener('click', () => stressNode(btn.dataset.stress));
  });
  grid.querySelectorAll('[data-release]').forEach(btn => {
    btn.addEventListener('click', () => releaseNode(btn.dataset.release));
  });
}

function nodeCardHTML(node, stressedNodes) {
  const stressed = stressedNodes.includes(node.node_name);
  const cpu    = Math.round((node.features.cpu_mean  || 0) * 100);
  const mem    = Math.round((node.features.mem_mean  || 0) * 100);
  const load   = (node.features.load_mean || 0).toFixed(2);
  const score  = node.score;

  const scoreClass = score >= 70 ? 'high' : score >= 40 ? 'mid' : 'low';
  const cpuClass   = cpu <= 50   ? 'green' : cpu <= 75 ? 'yellow' : 'red';
  const memClass   = mem <= 60   ? 'green' : mem <= 80 ? 'yellow' : 'red';

  const statusTag = stressed
    ? ''   // handled by CSS ::before on .node-card--stressed
    : score >= 70
      ? '<span class="tag tag--green">Healthy</span>'
      : score >= 40
        ? '<span class="tag tag--yellow">Moderate</span>'
        : '<span class="tag tag--red">High Risk</span>';

  return `
    <div class="card node-card ${stressed ? 'node-card--stressed' : ''}">
      <div class="node-card__header">
        <div>
          <div class="node-card__name">${esc(node.node_name)}</div>
        </div>
        <div class="flex flex-center gap-sm">
          ${statusTag}
          <div class="score-badge score-badge--${scoreClass}">${score}</div>
        </div>
      </div>

      <div class="node-card__metric">
        <div class="node-card__metric-label">
          <span>CPU</span>
          <span class="node-card__metric-val">${cpu}%</span>
        </div>
        <div class="progress">
          <div class="progress__bar progress__bar--${cpuClass}" style="width:${cpu}%"></div>
        </div>
      </div>

      <div class="node-card__metric">
        <div class="node-card__metric-label">
          <span>Memory</span>
          <span class="node-card__metric-val">${mem}%</span>
        </div>
        <div class="progress">
          <div class="progress__bar progress__bar--${memClass}" style="width:${mem}%"></div>
        </div>
      </div>

      <div class="node-card__metric">
        <div class="node-card__metric-label">
          <span>Load avg (1m)</span>
          <span class="node-card__metric-val">${load}</span>
        </div>
      </div>

      <div class="node-card__actions">
        ${stressed
          ? `<button class="btn btn--success" data-release="${esc(node.node_name)}">Release Stress</button>`
          : `<button class="btn btn--danger"  data-stress="${esc(node.node_name)}">Stress Node</button>`
        }
      </div>
    </div>
  `;
}

// ---------------------------------------------------------------------------
// Pod placement map
// ---------------------------------------------------------------------------

function renderPodMap(pods) {
  const container = document.getElementById('pod-map');
  if (!pods || pods.length === 0) {
    container.innerHTML = '<p class="text-muted text-sm">No pods detected (running outside cluster?)</p>';
    return;
  }

  // Group pods by node
  const byNode = {};
  pods.forEach(pod => {
    const n = pod.nodeName || 'pending';
    if (!byNode[n]) byNode[n] = [];
    byNode[n].push(pod);
  });

  const currentPodNames = new Set(pods.map(p => p.name));

  container.innerHTML = Object.entries(byNode).map(([node, nodePods]) => `
    <div class="pod-map__node-group">
      <div class="pod-map__node-label">${esc(node)}</div>
      <div class="pod-map__pods">
        ${nodePods.map(pod => {
          const isNew = !lastPodNames.has(pod.name);
          const cls   = isNew ? 'pod-chip pod-chip--new' : pod.ready ? 'pod-chip pod-chip--ready' : 'pod-chip';
          return `<span class="${cls}" title="${esc(pod.phase)}">${esc(pod.name.split('-').slice(-2).join('-'))}</span>`;
        }).join('')}
      </div>
    </div>
  `).join('');

  lastPodNames = currentPodNames;
}

// ---------------------------------------------------------------------------
// Flood stats
// ---------------------------------------------------------------------------

function renderFloodStats(flood) {
  if (!flood) return;

  const startBtn = document.getElementById('flood-start-btn');
  const stopBtn  = document.getElementById('flood-stop-btn');
  const activeBar = document.getElementById('flood-active-bar');
  const statsDiv  = document.getElementById('flood-stats');

  if (flood.active) {
    startBtn.disabled = true;
    stopBtn.disabled  = false;
    activeBar.style.display = 'flex';
    statsDiv.style.display  = 'grid';

    document.getElementById('stat-sent').textContent = fmtNum(flood.sent);
    document.getElementById('stat-rps').textContent  = fmtNum(flood.actualRps);
    document.getElementById('stat-ok').textContent   = fmtNum(flood.success);
    document.getElementById('stat-err').textContent  = fmtNum(flood.failed);
  } else {
    startBtn.disabled = false;
    stopBtn.disabled  = true;
    activeBar.style.display = 'none';
    statsDiv.style.display  = flood.sent > 0 ? 'grid' : 'none';
  }
}

// ---------------------------------------------------------------------------
// Admin API calls
// ---------------------------------------------------------------------------

async function adminPost(path, body = {}) {
  const r = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return r.json();
}

async function stressNode(nodeName) {
  const btn = document.querySelector(`[data-stress="${nodeName}"]`);
  if (btn) { btn.disabled = true; btn.textContent = 'Stressing...'; }
  try {
    await adminPost('/admin/stress', { nodeName });
  } catch (e) {
    alert('Failed to stress node: ' + e.message);
  }
}

async function releaseNode(nodeName) {
  const btn = document.querySelector(`[data-release="${nodeName}"]`);
  if (btn) { btn.disabled = true; btn.textContent = 'Releasing...'; }
  try {
    await adminPost('/admin/release', { nodeName });
  } catch (e) {
    alert('Failed to release node: ' + e.message);
  }
}

document.getElementById('flood-start-btn').addEventListener('click', async () => {
  const rps  = parseInt(document.getElementById('rps-input').value, 10);
  const conc = parseInt(document.getElementById('conc-input').value, 10);
  await adminPost('/admin/flood/start', { rps, concurrency: conc });
});

document.getElementById('flood-stop-btn').addEventListener('click', async () => {
  await adminPost('/admin/flood/stop');
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function esc(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function fmtNum(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(n);
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

connectSSE();
