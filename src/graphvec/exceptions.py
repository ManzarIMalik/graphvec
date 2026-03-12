"""Custom exception hierarchy for graphvec.

All exceptions derive from :class:`GraphVecError` so callers can catch
the entire family with a single ``except GraphVecError`` clause while
still being able to handle specific cases individually.
"""

from __future__ import annotations


class GraphVecError(Exception):
    """Base exception for all graphvec errors."""


class NodeNotFound(GraphVecError):
    """Raised when a requested node does not exist in the graph."""

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        super().__init__(f"Node not found: {node_id!r}")


class EdgeNotFound(GraphVecError):
    """Raised when a requested edge does not exist in the graph."""

    def __init__(self, edge_id: str) -> None:
        self.edge_id = edge_id
        super().__init__(f"Edge not found: {edge_id!r}")


class EmbeddingNotFound(GraphVecError):
    """Raised when no embedding is stored for a given node."""

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        super().__init__(f"No embedding stored for node: {node_id!r}")


class StorageError(GraphVecError):
    """Raised when the storage backend encounters an unrecoverable error."""


class CollectionNotFound(GraphVecError):
    """Raised when a requested collection does not exist."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Collection not found: {name!r}")


class IndexError(GraphVecError):  # noqa: A001 — intentional shadowing
    """Raised when index creation or deletion fails."""
