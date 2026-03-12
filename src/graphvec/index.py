"""Index management helpers.

Provides the :class:`IndexManager` which wraps the backend's index
primitives and exposes them through the public :class:`~graphvec.graph.Graph`
API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from graphvec.exceptions import StorageError

if TYPE_CHECKING:
    from graphvec.storage.base import StorageBackend


class IndexManager:
    """Create, list, and drop indexes for a single graph collection.

    Args:
        backend: The storage backend.
        collection: The collection namespace.
    """

    def __init__(self, backend: StorageBackend, collection: str) -> None:
        self._backend = backend
        self._collection = collection

    def create_index(self, target: str, field: str) -> None:
        """Create an index on *field* for *target* (``'nodes'`` or ``'edges'``).

        Supports plain column names (e.g. ``"label"``) and JSON property
        paths (e.g. ``"properties.confidence"``).

        Args:
            target: ``"nodes"`` or ``"edges"``.
            field: Column or JSON property path to index.

        Raises:
            StorageError: If the target is unknown or the SQL fails.
        """
        if target not in ("nodes", "edges"):
            raise StorageError(
                f"Invalid index target {target!r}. Must be 'nodes' or 'edges'."
            )
        self._backend.create_index(self._collection, target, field)

    def drop_index(self, target: str, field: str) -> None:
        """Remove an index.

        Args:
            target: ``"nodes"`` or ``"edges"``.
            field: The field that was indexed.
        """
        self._backend.drop_index(self._collection, target, field)

    def list_indexes(self) -> list[dict[str, Any]]:
        """Return all index descriptors for this collection.

        Returns:
            List of ``{"name": ..., "target": ..., "field": ...}`` dicts.
        """
        return self._backend.list_indexes(self._collection)
