"use strict";

/**
 * db.js — lightweight file-backed todo store.
 *
 * This avoids native database bindings while still retaining state across
 * restarts and, when backed by a shared Kubernetes volume, across pods.
 */

const fs = require("fs");
const path = require("path");

const DB_PATH = process.env.DB_PATH || "/data/tasks.json";
let state = { todos: [], nextId: 1 };
let loaded = false;

function ensureStore() {
  if (loaded) return;

  const dir = path.dirname(DB_PATH);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  if (fs.existsSync(DB_PATH)) {
    try {
      const raw = fs.readFileSync(DB_PATH, "utf8");
      if (raw.trim()) {
        const parsed = JSON.parse(raw);
        state = {
          todos: Array.isArray(parsed.todos) ? parsed.todos : [],
          nextId: typeof parsed.nextId === "number" ? parsed.nextId : 1,
        };
      }
    } catch (_) {
      state = { todos: [], nextId: 1 };
    }
  } else {
    writeState();
  }

  loaded = true;
}

function writeState() {
  const tmpPath = `${DB_PATH}.tmp`;
  fs.writeFileSync(tmpPath, JSON.stringify(state, null, 2));
  fs.renameSync(tmpPath, DB_PATH);
}

function listTodos() {
  ensureStore();
  return [...state.todos].sort((a, b) => b.id - a.id);
}

function getTodo(id) {
  ensureStore();
  return state.todos.find((todo) => todo.id === id) || null;
}

function createTodo(title) {
  ensureStore();
  const todo = {
    id: state.nextId++,
    title,
    done: 0,
    created_at: new Date().toISOString(),
  };
  state.todos.push(todo);
  writeState();
  return todo;
}

function updateTodo(id, { title, done }) {
  ensureStore();
  const existing = getTodo(id);
  if (!existing) return null;

  existing.title = title ?? existing.title;
  existing.done = done ?? existing.done;
  writeState();
  return existing;
}

function deleteTodo(id) {
  ensureStore();
  const idx = state.todos.findIndex((todo) => todo.id === id);
  if (idx >= 0) {
    state.todos.splice(idx, 1);
    writeState();
  }
}

function todoCount() {
  ensureStore();
  return state.todos.length;
}

module.exports = { listTodos, getTodo, createTodo, updateTodo, deleteTodo, todoCount };
