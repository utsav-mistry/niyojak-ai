"use strict";

/**
 * server.js — Express REST API and admin control backend.
 *
 * Routes:
 *   Public To-Do API:
 *     GET    /api/todos
 *     POST   /api/todos
 *     PUT    /api/todos/:id
 *     DELETE /api/todos/:id
 *
 *   Admin control API (served at /admin/*):
 *     GET  /admin/status         current node scores, pod map, flood status
 *     POST /admin/stress         { nodeName } -> launch niyojak-saturate Job
 *     POST /admin/release        { nodeName } -> delete saturate Job
 *     POST /admin/flood/start    { rps, concurrency } -> begin traffic flood
 *     POST /admin/flood/stop     -> halt flood
 *     GET  /admin/events         SSE stream — pushes status every 2s
 *
 *   Static files:
 *     GET  /          -> public/index.html
 *     GET  /admin     -> public/admin.html
 */

const path = require("path");
const express = require("express");
const k8s = require("@kubernetes/client-node");

const db = require("./db");
const stress = require("./stress_controller");
const flood = require("./traffic_generator");

const app = express();
const PORT = parseInt(process.env.PORT || "3000", 10);
const AI_SERVICE_URL = process.env.AI_SERVICE_URL || "http://niyojak-aiservice:8000";
const BASE_URL = process.env.BASE_URL || `http://localhost:${PORT}`;

// Kubernetes client for pod listing (pod placement map).
const kc = new k8s.KubeConfig();
kc.loadFromDefault();
const coreApi = kc.makeApiClient(k8s.CoreV1Api);

app.use(express.json());

// Serve static frontend files.
app.use(express.static(path.join(__dirname, "..", "public")));

// SPA fallback for /admin route.
app.get("/admin", (_req, res) => {
  res.sendFile(path.join(__dirname, "..", "public", "admin.html"));
});

// ---------------------------------------------------------------------------
// To-Do CRUD API
// ---------------------------------------------------------------------------

app.get("/api/todos", (_req, res) => {
  res.json(db.listTodos());
});

app.post("/api/todos", (req, res) => {
  const { title } = req.body;
  if (!title || typeof title !== "string" || title.trim() === "") {
    return res.status(400).json({ error: "title is required" });
  }
  const todo = db.createTodo(title.trim());
  res.status(201).json(todo);
});

app.put("/api/todos/:id", (req, res) => {
  const id = parseInt(req.params.id, 10);
  const existing = db.getTodo(id);
  if (!existing) return res.status(404).json({ error: "not found" });

  const { title = existing.title, done = existing.done } = req.body;
  const updated = db.updateTodo(id, { title, done });
  res.json(updated);
});

app.delete("/api/todos/:id", (req, res) => {
  const id = parseInt(req.params.id, 10);
  db.deleteTodo(id);
  res.status(204).end();
});

// ---------------------------------------------------------------------------
// Admin status helper — fetches node scores from AI service and pod list
// ---------------------------------------------------------------------------

async function fetchStatus() {
  // Node AI scores
  let nodes = [];
  try {
    const r = await fetch(`${AI_SERVICE_URL}/nodes`, { signal: AbortSignal.timeout(3000) });
    if (r.ok) nodes = await r.json();
  } catch (_) {}

  // Pod placement map
  let pods = [];
  try {
    const res = await coreApi.listPodForAllNamespaces(
      undefined, undefined, undefined, "app=todo-app"
    );
    pods = res.body.items.map((p) => ({
      name:      p.metadata.name,
      namespace: p.metadata.namespace,
      nodeName:  p.spec.nodeName || "pending",
      phase:     p.status.phase,
      ready:     (p.status.containerStatuses || []).some((c) => c.ready),
    }));
  } catch (_) {}

  // Active stress jobs with profile info and countdown
  let stressJobs = [];
  try {
    stressJobs = await stress.activeStressJobs();
  } catch (_) {}

  return {
    nodes,
    pods,
    stressJobs,                                          // full job info for countdown
    stressedNodes: stressJobs.map(j => j.nodeName),     // backward-compat flat list
    flood:     flood.getStats(),
    todoCount: db.todoCount(),
    ts: Date.now(),
  };
}

// ---------------------------------------------------------------------------
// Admin API
// ---------------------------------------------------------------------------

app.get("/admin/status", async (_req, res) => {
  try {
    res.json(await fetchStatus());
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post("/admin/stress", async (req, res) => {
  const { nodeName, profile = "moderate" } = req.body;
  if (!nodeName) return res.status(400).json({ error: "nodeName required" });
  try {
    const result = await stress.stressNode(nodeName, profile);
    res.json(result);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get("/admin/profiles", (_req, res) => {
  res.json(stress.stressProfiles());
});

app.post("/admin/release", async (req, res) => {
  const { nodeName } = req.body;
  if (!nodeName) return res.status(400).json({ error: "nodeName required" });
  try {
    const result = await stress.releaseNode(nodeName);
    res.json(result);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post("/admin/flood/start", (req, res) => {
  const rps = parseInt(req.body.rps || "200", 10);
  const concurrency = parseInt(req.body.concurrency || "30", 10);
  const result = flood.start({ baseUrl: BASE_URL, rps, concurrency });
  res.json(result);
});

app.post("/admin/flood/stop", (_req, res) => {
  res.json(flood.stop());
});

// ---------------------------------------------------------------------------
// SSE stream — pushes a status update to all /admin/events subscribers every 2s
// ---------------------------------------------------------------------------

const sseClients = new Set();

app.get("/admin/events", (req, res) => {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders();

  sseClients.add(res);

  req.on("close", () => {
    sseClients.delete(res);
  });
});

// Push status to all SSE subscribers every 2 seconds.
setInterval(async () => {
  if (sseClients.size === 0) return;
  try {
    const status = await fetchStatus();
    const data = `data: ${JSON.stringify(status)}\n\n`;
    for (const client of sseClients) {
      client.write(data);
    }
  } catch (_) {}
}, 2000);

// ---------------------------------------------------------------------------
// Health check (for Kubernetes liveness probe)
// ---------------------------------------------------------------------------

app.get("/health", (_req, res) => {
  res.json({ status: "ok", todos: db.todoCount() });
});

// ---------------------------------------------------------------------------
// Start server
// ---------------------------------------------------------------------------

app.listen(PORT, () => {
  console.log(`[niyojak-todo] server running on port ${PORT}`);
  console.log(`[niyojak-todo] AI service: ${AI_SERVICE_URL}`);
});
