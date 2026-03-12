"""Abstract base class for graphvec storage backends.

Implementing this interface is all that is required to provide an
alternative persistence layer (LevelDB, PostgreSQL, DuckDB, …) without
touching any other part of the public API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class StorageBackend(ABC):
    """Protocol that every storage backend must satisfy.

    All methods operate within a single *collection* (an isolated named
    graph namespace).  The ``collection`` parameter is always a plain
    string identifier.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def open(self) -> None:
        """Open / initialise the backend (create schema if needed)."""

    @abstractmethod
    def close(self) -> None:
        """Flush and release all resources."""

    # ------------------------------------------------------------------
    # Low-level transaction support
    # ------------------------------------------------------------------

    @abstractmethod
    def begin(self) -> None:
        """Begin an explicit transaction."""

    @abstractmethod
    def commit(self) -> None:
        """Commit the current transaction."""

    @abstractmethod
    def rollback(self) -> None:
        """Rollback the current transaction."""

    @abstractmethod
    def autocommit(self) -> None:
        """Commit only when NOT inside an explicit transaction.

        Called by individual graph operations so that they auto-commit in
        standalone use but remain part of the caller's transaction when
        :py:meth:`begin` has been called.
        """

    # ------------------------------------------------------------------
    # Node persistence
    # ------------------------------------------------------------------

    @abstractmethod
    def insert_node(self, collection: str, row: dict[str, Any]) -> None:
        """Insert a node row.  *row* must match the nodes schema."""

    @abstractmethod
    def fetch_node(self, collection: str, node_id: str) -> dict[str, Any] | None:
        """Return a raw node row or ``None`` if not found."""

    @abstractmethod
    def fetch_nodes_by_ids(
        self, collection: str, node_ids: list[str]
    ) -> list[dict[str, Any]]:
        """Return rows for all *node_ids* in a single query (bulk fetch)."""

    @abstractmethod
    def update_node(self, collection: str, node_id: str, updates: dict[str, Any]) -> None:
        """Merge *updates* into the node's properties and timestamps."""

    @abstractmethod
    def delete_node(self, collection: str, node_id: str) -> None:
        """Delete a node and cascade to its edges and embeddings."""

    @abstractmethod
    def query_nodes(
        self,
        collection: str,
        label: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return all nodes matching optional *label* and property *filters*."""

    @abstractmethod
    def count_nodes(self, collection: str) -> int:
        """Return total number of nodes without loading rows into Python."""

    # ------------------------------------------------------------------
    # Edge persistence
    # ------------------------------------------------------------------

    @abstractmethod
    def insert_edge(self, collection: str, row: dict[str, Any]) -> None:
        """Insert an edge row."""

    @abstractmethod
    def fetch_edge(self, collection: str, edge_id: str) -> dict[str, Any] | None:
        """Return a raw edge row or ``None`` if not found."""

    @abstractmethod
    def update_edge(self, collection: str, edge_id: str, updates: dict[str, Any]) -> None:
        """Merge *updates* into the edge's properties and timestamps."""

    @abstractmethod
    def delete_edge(self, collection: str, edge_id: str) -> None:
        """Delete an edge by id."""

    @abstractmethod
    def query_edges(
        self,
        collection: str,
        label: str | None = None,
        src: str | None = None,
        dst: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return all edges matching the provided filters."""

    @abstractmethod
    def count_edges(
        self,
        collection: str,
        label: str | None = None,
        src: str | None = None,
        dst: str | None = None,
    ) -> int:
        """Return edge count without loading rows into Python."""

    @abstractmethod
    def has_edge(
        self,
        collection: str,
        src: str,
        dst: str,
        label: str | None = None,
    ) -> bool:
        """Return ``True`` if at least one matching edge exists (``LIMIT 1``)."""

    # ------------------------------------------------------------------
    # Embedding persistence
    # ------------------------------------------------------------------

    @abstractmethod
    def insert_embedding(self, collection: str, row: dict[str, Any]) -> None:
        """Insert or replace an embedding row."""

    @abstractmethod
    def fetch_embedding(self, collection: str, node_id: str) -> dict[str, Any] | None:
        """Return a raw embedding row or ``None`` if not found."""

    @abstractmethod
    def fetch_all_embeddings(self, collection: str) -> list[dict[str, Any]]:
        """Return every embedding row for the collection."""

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    @abstractmethod
    def create_index(self, collection: str, target: str, field: str) -> None:
        """Create an index on *field* of *target* (``'nodes'`` or ``'edges'``)."""

    @abstractmethod
    def drop_index(self, collection: str, target: str, field: str) -> None:
        """Remove an index."""

    @abstractmethod
    def list_indexes(self, collection: str) -> list[dict[str, Any]]:
        """Return all index descriptors for the collection."""

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    @abstractmethod
    def list_collections(self) -> list[str]:
        """Return the names of all collections in this database."""

    @abstractmethod
    def drop_collection(self, collection: str) -> None:
        """Permanently delete all data for *collection*."""
