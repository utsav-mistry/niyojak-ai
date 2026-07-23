"use strict";

/**
 * traffic_generator.js — In-process HTTP request flood engine.
 *
 * Fires concurrent HTTP requests against the To-Do App's own API endpoints
 * to drive CPU load on the running pods. When load exceeds the HPA threshold,
 * Kubernetes automatically creates new pod replicas which niyojak-scheduler
 * then places on the healthiest available node.
 *
 * This runs inside the same Express process — no external binary is needed.
 * The flood is managed as a set of concurrent interval loops that can be
 * started and stopped on demand from the /admin API.
 */

const http = require("http");
const https = require("https");

let _floodActive = false;
let _workers = []; // array of setInterval handles
let _stats = { sent: 0, success: 0, failed: 0, startedAt: null };

/**
 * start begins flooding the given baseUrl at approximately rps
 * requests per second using concurrency parallel workers.
 * Calling start while a flood is already running is a no-op.
 */
function start({ baseUrl, rps = 200, concurrency = 30 }) {
  if (_floodActive) return { ok: false, reason: "flood already active" };

  _floodActive = true;
  _stats = { sent: 0, success: 0, failed: 0, startedAt: Date.now() };

  const intervalMs = Math.max(1, Math.round((concurrency / rps) * 1000));
  const lib = baseUrl.startsWith("https") ? https : http;

  for (let i = 0; i < concurrency; i++) {
    const handle = setInterval(() => {
      if (!_floodActive) {
        clearInterval(handle);
        return;
      }
      _doRequest(lib, baseUrl);
    }, intervalMs);
    _workers.push(handle);
  }

  return { ok: true, rps, concurrency, intervalMs };
}

/**
 * stop halts all flood workers.
 */
function stop() {
  if (!_floodActive) return { ok: false, reason: "no flood active" };
  _floodActive = false;
  _workers.forEach(clearInterval);
  _workers = [];
  return { ok: true, stats: getStats() };
}

/**
 * isActive returns true while a flood is running.
 */
function isActive() {
  return _floodActive;
}

/**
 * getStats returns current counters for the live stats panel.
 */
function getStats() {
  const elapsed = _stats.startedAt
    ? (Date.now() - _stats.startedAt) / 1000
    : 0;
  return {
    active: _floodActive,
    sent: _stats.sent,
    success: _stats.success,
    failed: _stats.failed,
    elapsedSec: Math.round(elapsed),
    actualRps: elapsed > 0 ? Math.round(_stats.sent / elapsed) : 0,
  };
}

/**
 * _doRequest fires a single HTTP request — randomly chosen between
 * GET /api/todos (60%), POST /api/todos (30%), DELETE /api/todos/1 (10%).
 */
function _doRequest(lib, baseUrl) {
  _stats.sent++;
  const roll = Math.random();

  let options;
  let body = null;

  if (roll < 0.6) {
    options = _parseUrl(lib, baseUrl, "GET", "/api/todos");
  } else if (roll < 0.9) {
    body = JSON.stringify({ title: `load-${Date.now()}` });
    options = _parseUrl(lib, baseUrl, "POST", "/api/todos");
    options.headers = {
      "Content-Type": "application/json",
      "Content-Length": Buffer.byteLength(body),
    };
  } else {
    options = _parseUrl(lib, baseUrl, "DELETE", "/api/todos/1");
  }

  const req = lib.request(options, (res) => {
    res.resume(); // drain the response body so the socket is reused
    if (res.statusCode < 500) {
      _stats.success++;
    } else {
      _stats.failed++;
    }
  });

  req.on("error", () => {
    _stats.failed++;
  });

  req.setTimeout(4000, () => {
    req.destroy();
    _stats.failed++;
  });

  if (body) {
    req.write(body);
  }
  req.end();
}

function _parseUrl(lib, baseUrl, method, path) {
  const u = new URL(baseUrl);
  return {
    hostname: u.hostname,
    port: u.port || (lib === https ? 443 : 80),
    path,
    method,
  };
}

module.exports = { start, stop, isActive, getStats };
