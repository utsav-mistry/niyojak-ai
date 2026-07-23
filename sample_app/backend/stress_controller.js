"use strict";

/**
 * stress_controller.js — Controlled Kubernetes Job launcher for niyojak-saturate.
 *
 * Stress is applied via pre-defined profiles rather than raw numbers so the
 * presenter can never accidentally take a node offline during a demo.
 * Every Job has a hard activeDeadlineSeconds cap — Kubernetes auto-terminates it
 * even if the presenter forgets to click "Release Stress".
 *
 * Profiles:
 *   light    — 40% CPU, 128 MB RAM, max 5 minutes
 *   moderate — 65% CPU, 256 MB RAM, max 5 minutes
 *   heavy    — 85% CPU, 512 MB RAM, max 5 minutes
 */

const k8s = require("@kubernetes/client-node");

const NAMESPACE = process.env.NIYOJAK_NAMESPACE || "niyojak-system";
const SATURATE_IMAGE = process.env.SATURATE_IMAGE || "ghcr.io/niyojak/niyojak-saturate:latest";
const SAFE_STRESS_MODE = process.env.NIYOJAK_SAFE_STRESS_MODE !== "false";

// Hard ceiling on how long any stress Job can run — Kubernetes enforces this
// even if the API call to releaseNode is never made.
const MAX_STRESS_SECONDS = parseInt(
  process.env.MAX_STRESS_SEC || (SAFE_STRESS_MODE ? "180" : "300"),
  10
);
const HARD_MODE_MAX_SECONDS = 60;

// Predefined stress profiles — presenter picks one, never raw percentages.
const PROFILES = {
  light: {
    cpu: "40",    // % of one vCPU
    mem_mb: "128",
    label: "Light (40% CPU, 128 MB)",
  },
  moderate: {
    cpu: "65",
    mem_mb: "256",
    label: "Moderate (65% CPU, 256 MB)",
  },
  heavy: {
    cpu: "85",
    mem_mb: "512",
    label: "Heavy (85% CPU, 512 MB)",
  },
};

const kc = new k8s.KubeConfig();
kc.loadFromDefault();
const batchApi = kc.makeApiClient(k8s.BatchV1Api);

/**
 * Deterministic Job name per node so we can delete it by name.
 */
function jobName(nodeName) {
  return `niyojak-saturate-${nodeName.replace(/[^a-z0-9-]/g, "-").toLowerCase()}`;
}

/**
 * stressNode creates a Kubernetes Job that runs niyojak-saturate on nodeName.
 *
 * @param {string} nodeName   - Kubernetes node hostname
 * @param {string} profile    - one of: "light" | "moderate" | "heavy"
 * @returns {{ ok: boolean, job: string, profile: string, expiresInSec: number }}
 */
async function stressNode(nodeName, profile = "light") {
  const config = PROFILES[profile];
  if (!config) {
    throw new Error(`Unknown stress profile "${profile}". Valid: ${Object.keys(PROFILES).join(", ")}`);
  }

  if (SAFE_STRESS_MODE && profile === "heavy") {
    throw new Error("Heavy stress is disabled in safe mode. Use light or moderate instead.");
  }

  const effectiveDuration = profile === "heavy" ? HARD_MODE_MAX_SECONDS : MAX_STRESS_SECONDS;
  const name = jobName(nodeName);

  const job = {
    apiVersion: "batch/v1",
    kind: "Job",
    metadata: {
      name,
      namespace: NAMESPACE,
      labels: {
        "niyojak/role": "saturate",
        "niyojak/node": nodeName,
        "niyojak/profile": profile,
      },
      annotations: {
        "niyojak/started-at": new Date().toISOString(),
        "niyojak/profile-label": config.label,
      },
    },
    spec: {
      // Hard auto-terminate — even if release is never called.
      activeDeadlineSeconds: effectiveDuration,
      backoffLimit: 0,
      template: {
        metadata: { labels: { "niyojak/role": "saturate" } },
        spec: {
          restartPolicy: "Never",
          nodeSelector: { "kubernetes.io/hostname": nodeName },
          tolerations: [{ operator: "Exists" }],
          containers: [
            {
              name: "saturate",
              image: SATURATE_IMAGE,
              args: [
                `--cpu=${config.cpu}`,
                `--mem=${config.mem_mb}`,
                `--duration=${effectiveDuration}s`,
              ],
              resources: {
                requests: { cpu: "100m", memory: "64Mi" },
                // Limit RAM to the profile ceiling + headroom to avoid OOMKill
                limits: { memory: `${parseInt(config.mem_mb, 10) + 64}Mi` },
              },
            },
          ],
        },
      },
    },
  };

  try {
    await batchApi.createNamespacedJob(NAMESPACE, job);
    return { ok: true, job: name, profile: config.label, expiresInSec: effectiveDuration, safeMode: SAFE_STRESS_MODE, hardModeCapSec: HARD_MODE_MAX_SECONDS };
  } catch (err) {
    if (err.statusCode === 409) {
      return { ok: true, job: name, note: "already running", profile: config.label, expiresInSec: effectiveDuration, safeMode: SAFE_STRESS_MODE, hardModeCapSec: HARD_MODE_MAX_SECONDS };
    }
    throw err;
  }
}

/**
 * releaseNode deletes the stress Job immediately, stopping niyojak-saturate.
 */
async function releaseNode(nodeName) {
  const name = jobName(nodeName);
  try {
    await batchApi.deleteNamespacedJob(
      name, NAMESPACE, undefined, undefined, undefined, undefined, "Foreground"
    );
    return { ok: true };
  } catch (err) {
    if (err.statusCode === 404) {
      return { ok: true, note: "job not found (already released or expired)" };
    }
    throw err;
  }
}

/**
 * activeStressJobs returns info about all currently running stress Jobs.
 * @returns {{ nodeName: string, profile: string, startedAt: string }[]}
 */
async function activeStressJobs() {
  const res = await batchApi.listNamespacedJob(
    NAMESPACE, undefined, undefined, undefined, undefined,
    "niyojak/role=saturate"
  );
  return res.body.items
    .filter(j => !j.status.completionTime)  // exclude finished/expired Jobs
    .map(j => ({
      nodeName: j.metadata.labels["niyojak/node"],
      profile: j.metadata.annotations?.["niyojak/profile-label"] || "unknown",
      startedAt: j.metadata.annotations?.["niyojak/started-at"] || "",
      expiresInSec: MAX_STRESS_SECONDS,
    }));
}

/**
 * stressProfiles returns the available profile options for the admin UI.
 */
function stressProfiles() {
  return Object.entries(PROFILES).map(([key, val]) => ({ key, label: val.label }));
}

module.exports = { stressNode, releaseNode, activeStressJobs, stressProfiles };
