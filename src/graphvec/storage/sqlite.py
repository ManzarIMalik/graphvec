"""SQLite storage backend for graphvec.

Uses Python's stdlib ``sqlite3`` module — zero additional dependencies.
WAL mode is enabled for concurrent reads.  All graph data lives in a
single ``.db`` file; collections are isolated via table-name prefixing
(``<collection>_nodes``, ``<collection>_edges``, …).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any

from graphvec.exceptions import StorageError
from graphvec.storage.base import StorageBackend


def _safe_name(collection: str) -> str:
    """Validate collection name and return a safe SQL identifier prefix.

    Only ASCII alphanumeric characters and underscores are allowed so
    the name can be used directly inside SQL identifiers (table names)
    without the risk of SQL injection.
    """
    if not collection.replace("_", "").isalnum():
        raise StorageError(
            f"Collection name {collection!r} is invalid. "
            "Use only ASCII letters, digits, and underscores."
        )
    return collection


class SQLiteBackend(StorageBackend):
    """Persist a graphvec database to a single SQLite file.

    Args:
        path: Filesystem path to the ``.db`` file, or ``":memory:"`` for
              a transient in-memory database (useful for tests).
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._local = threading.local()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _conn(self) -> sqlite3.Connection:
        """Return the thread-local SQLite connection, creating it if needed."""
        if not getattr(self._local, "conn", None):
            conn = sqlite3.connect(self._path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn  # type: ignore[return-value]

    def _exe(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        try:
            return self._conn.execute(sql, params)
        except sqlite3.Error as exc:
            raise StorageError(str(exc)) from exc

    def _create_collection_tables(self, collection: str) -> None:
        """Idempotently create the four tables for *collection*."""
        c = _safe_name(collection)
        stmts = [
            f"""
            CREATE TABLE IF NOT EXISTS "{c}_nodes" (
                id          TEXT PRIMARY KEY,
                label       TEXT,
                properties  TEXT DEFAULT '{{}}',
                created_at  REAL,
                updated_at  REAL
            )""",
            f"""
            CREATE TABLE IF NOT EXISTS "{c}_edges" (
                id          TEXT PRIMARY KEY,
                src         TEXT REFERENCES "{c}_nodes"(id) ON DELETE CASCADE,
                dst         TEXT REFERENCES "{c}_nodes"(id) ON DELETE CASCADE,
                label       TEXT,
                properties  TEXT DEFAULT '{{}}',
                weight      REAL DEFAULT 1.0,
                created_at  REAL,
                updated_at  REAL
            )""",
            f"""
            CREATE TABLE IF NOT EXISTS "{c}_embeddings" (
                node_id     TEXT PRIMARY KEY
                                REFERENCES "{c}_nodes"(id) ON DELETE CASCADE,
                vector      BLOB,
                model       TEXT,
                dimensions  INTEGER,
                created_at  REAL
            )""",
            f"""
            CREATE TABLE IF NOT EXISTS "{c}_indexes" (
                name        TEXT PRIMARY KEY,
                target      TEXT,
                field       TEXT
            )""",
            # Track which collections exist
            """
            CREATE TABLE IF NOT EXISTS _collections (
                name TEXT PRIMARY KEY
            )""",
            "INSERT OR IGNORE INTO _collections(name) VALUES (?)",
        ]
        try:
            for stmt in stmts[:-1]:
                self._conn.execute(stmt)
            self._conn.execute(stmts[-1], (collection,))
            self._conn.commit()
        except sqlite3.Error as exc:
            raise StorageError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Initialise the connection and ensure the collections registry exists."""
        try:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS _collections (name TEXT PRIMARY KEY)"
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise StorageError(str(exc)) from exc

    def close(self) -> None:
        """Close the thread-local connection if open."""
        conn = getattr(self._local, "conn", None)
        if conn:
            conn.close()
            self._local.conn = None

    # ------------------------------------------------------------------
    # Transaction support
    # ------------------------------------------------------------------

    def begin(self) -> None:
        """Begin an explicit deferred transaction."""
        self._exe("BEGIN")
        self._local.in_transaction = True

    def commit(self) -> None:
        """Commit the current transaction."""
        try:
            self._conn.commit()
        except sqlite3.Error as exc:
            raise StorageError(str(exc)) from exc
        self._local.in_transaction = False

    def rollback(self) -> None:
        """Rollback the current transaction."""
        try:
            self._conn.rollback()
        except sqlite3.Error as exc:
            raise StorageError(str(exc)) from exc
        self._local.in_transaction = False

    def autocommit(self) -> None:
        """Commit only when NOT inside an explicit :py:meth:`begin` block.

        Individual graph operations call this so they are auto-committed in
        standalone use but remain part of the caller's transaction when
        used inside a ``with g.transaction():`` block.
        """
        if not getattr(self._local, "in_transaction", False):
            self.commit()

    # ------------------------------------------------------------------
    # Internal: ensure collection tables exist
    # ------------------------------------------------------------------

    def _ensure(self, collection: str) -> str:
        """Ensure collection tables exist and return the safe name.

        Results are cached per thread so the ``CREATE TABLE IF NOT EXISTS``
        statements are only executed once per collection per thread.
        """
        c = _safe_name(collection)
        initialized: set[str] = getattr(self._local, "initialized_collections", set())
        if c not in initialized:
            self._create_collection_tables(c)
            initialized.add(c)
            self._local.initialized_collections = initialized
        return c

    # ------------------------------------------------------------------
    # Node persistence
    # ------------------------------------------------------------------

    def insert_node(self, collection: str, row: dict[str, Any]) -> None:
        c = self._ensure(collection)
        props = json.dumps(row.get("properties", {}))
        self._exe(
            f'INSERT INTO "{c}_nodes" (id, label, properties, created_at, updated_at) '
            "VALUES (?, ?, ?, ?, ?)",
            (row["id"], row["label"], props, row["created_at"], row["updated_at"]),
        )

    def fetch_node(self, collection: str, node_id: str) -> dict[str, Any] | None:
        c = self._ensure(collection)
        row = self._exe(
            f'SELECT * FROM "{c}_nodes" WHERE id = ?', (node_id,)
        ).fetchone()
        return _node_row(row) if row else None

    def fetch_nodes_by_ids(
        self, collection: str, node_ids: list[str]
    ) -> list[dict[str, Any]]:
        if not node_ids:
            return []
        c = self._ensure(collection)
        placeholders = ",".join("?" * len(node_ids))
        rows = self._exe(
            f'SELECT * FROM "{c}_nodes" WHERE id IN ({placeholders})',
            tuple(node_ids),
        ).fetchall()
        return [_node_row(r) for r in rows]

    def update_node(self, collection: str, node_id: str, updates: dict[str, Any]) -> None:
        c = self._ensure(collection)
        existing = self.fetch_node(collection, node_id)
        if existing is None:
            return
        merged = {**existing["properties"], **updates.get("properties", updates)}
        # Remove internal keys that shouldn't land in properties
        for k in ("id", "label", "created_at", "updated_at", "properties"):
            merged.pop(k, None)
        self._exe(
            f'UPDATE "{c}_nodes" SET properties = ?, updated_at = ? WHERE id = ?',
            (json.dumps(merged), updates.get("updated_at", existing["updated_at"]), node_id),
        )

    def delete_node(self, collection: str, node_id: str) -> None:
        c = self._ensure(collection)
        self._exe(f'DELETE FROM "{c}_nodes" WHERE id = ?', (node_id,))

    def query_nodes(
        self,
        collection: str,
        label: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        c = self._ensure(collection)
        sql = f'SELECT * FROM "{c}_nodes"'
        params: list[Any] = []
        conditions: list[str] = []
        if label is not None:
            conditions.append("label = ?")
            params.append(label)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        rows = self._exe(sql, tuple(params)).fetchall()
        results = [_node_row(r) for r in rows]
        if filters:
            results = [
                r for r in results
                if all(r["properties"].get(k) == v for k, v in filters.items())
            ]
        return results

    # ------------------------------------------------------------------
    # Edge persistence
    # ------------------------------------------------------------------

    def insert_edge(self, collection: str, row: dict[str, Any]) -> None:
        c = self._ensure(collection)
        props = json.dumps(row.get("properties", {}))
        self._exe(
            f'INSERT INTO "{c}_edges" '
            "(id, src, dst, label, properties, weight, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row["id"],
                row["src"],
                row["dst"],
                row["label"],
                props,
                row.get("weight", 1.0),
                row["created_at"],
                row["updated_at"],
            ),
        )

    def fetch_edge(self, collection: str, edge_id: str) -> dict[str, Any] | None:
        c = self._ensure(collection)
        row = self._exe(
            f'SELECT * FROM "{c}_edges" WHERE id = ?', (edge_id,)
        ).fetchone()
        return _edge_row(row) if row else None

    def update_edge(self, collection: str, edge_id: str, updates: dict[str, Any]) -> None:
        c = self._ensure(collection)
        existing = self.fetch_edge(collection, edge_id)
        if existing is None:
            return
        merged = {**existing["properties"], **updates.get("properties", updates)}
        for k in ("id", "src", "dst", "label", "weight", "created_at", "updated_at", "properties"):
            merged.pop(k, None)
        weight = updates.get("weight", existing["weight"])
        self._exe(
            f'UPDATE "{c}_edges" SET properties = ?, weight = ?, updated_at = ? WHERE id = ?',
            (json.dumps(merged), weight, updates.get("updated_at", existing["updated_at"]), edge_id),
        )

    def delete_edge(self, collection: str, edge_id: str) -> None:
        c = self._ensure(collection)
        self._exe(f'DELETE FROM "{c}_edges" WHERE id = ?', (edge_id,))

    def query_edges(
        self,
        collection: str,
        label: str | None = None,
        src: str | None = None,
        dst: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        c = self._ensure(collection)
        sql = f'SELECT * FROM "{c}_edges"'
        params: list[Any] = []
        conditions: list[str] = []
        if label is not None:
            conditions.append("label = ?")
            params.append(label)
        if src is not None:
            conditions.append("src = ?")
            params.append(src)
        if dst is not None:
            conditions.append("dst = ?")
            params.append(dst)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        rows = self._exe(sql, tuple(params)).fetchall()
        results = [_edge_row(r) for r in rows]
        if filters:
            results = [
                r for r in results
                if all(r["properties"].get(k) == v for k, v in filters.items())
            ]
        return results

    # ------------------------------------------------------------------
    # Embedding persistence
    # ------------------------------------------------------------------

    def insert_embedding(self, collection: str, row: dict[str, Any]) -> None:
        c = self._ensure(collection)
        self._exe(
            f'INSERT OR REPLACE INTO "{c}_embeddings" '
            "(node_id, vector, model, dimensions, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (row["node_id"], row["vector"], row["model"], row["dimensions"], row["created_at"]),
        )

    def fetch_embedding(self, collection: str, node_id: str) -> dict[str, Any] | None:
        c = self._ensure(collection)
        row = self._exe(
            f'SELECT * FROM "{c}_embeddings" WHERE node_id = ?', (node_id,)
        ).fetchone()
        return dict(row) if row else None

    def fetch_all_embeddings(self, collection: str) -> list[dict[str, Any]]:
        c = self._ensure(collection)
        rows = self._exe(f'SELECT * FROM "{c}_embeddings"').fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def create_index(self, collection: str, target: str, field: str) -> None:
        c = self._ensure(collection)
        # Sanitise field to a safe SQL index name
        safe_field = field.replace(".", "_").replace(" ", "_")
        index_name = f"idx_{c}_{target}_{safe_field}"
        meta_name = f"{target}.{field}"

        if target == "nodes":
            table = f"{c}_nodes"
            if field == "label":
                col_expr = "label"
            elif field.startswith("properties."):
                prop_key = field[len("properties."):]
                col_expr = f"json_extract(properties, '$.{prop_key}')"
            else:
                col_expr = field
        elif target == "edges":
            table = f"{c}_edges"
            if field in ("label", "src", "dst", "weight"):
                col_expr = field
            elif field.startswith("properties."):
                prop_key = field[len("properties."):]
                col_expr = f"json_extract(properties, '$.{prop_key}')"
            else:
                col_expr = field
        else:
            raise StorageError(f"Unknown index target: {target!r}")

        try:
            self._conn.execute(
                f'CREATE INDEX IF NOT EXISTS "{index_name}" ON "{table}" ({col_expr})'
            )
            self._exe(
                f'INSERT OR IGNORE INTO "{c}_indexes" (name, target, field) VALUES (?, ?, ?)',
                (meta_name, target, field),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise StorageError(str(exc)) from exc

    def drop_index(self, collection: str, target: str, field: str) -> None:
        c = self._ensure(collection)
        safe_field = field.replace(".", "_").replace(" ", "_")
        index_name = f"idx_{c}_{target}_{safe_field}"
        meta_name = f"{target}.{field}"
        try:
            self._conn.execute(f'DROP INDEX IF EXISTS "{index_name}"')
            self._exe(
                f'DELETE FROM "{c}_indexes" WHERE name = ?', (meta_name,)
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise StorageError(str(exc)) from exc

    def list_indexes(self, collection: str) -> list[dict[str, Any]]:
        c = self._ensure(collection)
        rows = self._exe(f'SELECT * FROM "{c}_indexes"').fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def list_collections(self) -> list[str]:
        try:
            rows = self._exe("SELECT name FROM _collections ORDER BY name").fetchall()
            return [r["name"] for r in rows]
        except StorageError:
            return []

    def drop_collection(self, collection: str) -> None:
        c = _safe_name(collection)
        try:
            for suffix in ("embeddings", "edges", "nodes", "indexes"):
                self._conn.execute(f'DROP TABLE IF EXISTS "{c}_{suffix}"')
            self._conn.execute("DELETE FROM _collections WHERE name = ?", (collection,))
            self._conn.commit()
        except sqlite3.Error as exc:
            raise StorageError(str(exc)) from exc


# ------------------------------------------------------------------
# Row conversion helpers
# ------------------------------------------------------------------

def _node_row(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["properties"] = json.loads(d.get("properties") or "{}")
    return d


def _edge_row(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["properties"] = json.loads(d.get("properties") or "{}")
    return d
