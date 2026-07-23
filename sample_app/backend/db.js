"use strict";

/**
 * db.js — SQLite database driver with automatic schema initialisation.
 *
 * Uses better-sqlite3 (synchronous) so there is no callback complexity.
 * On every startup it creates tasks.db and the todos table if they do not
 * exist yet — no manual database setup is ever needed.
 */

const path = require("path");
const Database = require("better-sqlite3");

const DB_PATH = process.env.DB_PATH || path.join(__dirname, "..", "tasks.db");

let db;

function getDb() {
  if (!db) {
    db = new Database(DB_PATH);

    // Enable WAL mode for better read concurrency under load.
    db.pragma("journal_mode = WAL");
    db.pragma("foreign_keys = ON");

    // Auto-create schema on first run.
    db.exec(`
      CREATE TABLE IF NOT EXISTS todos (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        title      TEXT    NOT NULL,
        done       INTEGER NOT NULL DEFAULT 0,
        created_at TEXT    NOT NULL DEFAULT (datetime('now'))
      );
    `);
  }
  return db;
}

// --- Prepared statements (compiled once, reused across requests) ----------

function listTodos() {
  return getDb()
    .prepare("SELECT id, title, done, created_at FROM todos ORDER BY id DESC")
    .all();
}

function getTodo(id) {
  return getDb()
    .prepare("SELECT id, title, done, created_at FROM todos WHERE id = ?")
    .get(id);
}

function createTodo(title) {
  const result = getDb()
    .prepare("INSERT INTO todos (title) VALUES (?)")
    .run(title);
  return getTodo(result.lastInsertRowid);
}

function updateTodo(id, { title, done }) {
  getDb()
    .prepare("UPDATE todos SET title = ?, done = ? WHERE id = ?")
    .run(title, done ? 1 : 0, id);
  return getTodo(id);
}

function deleteTodo(id) {
  getDb().prepare("DELETE FROM todos WHERE id = ?").run(id);
}

function todoCount() {
  return getDb().prepare("SELECT COUNT(*) as n FROM todos").get().n;
}

module.exports = { listTodos, getTodo, createTodo, updateTodo, deleteTodo, todoCount };
