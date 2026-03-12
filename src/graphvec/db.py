"""Top-level GraphVec class and collection management.

This is the main entry point for the graphvec API.

Example::

    from graphvec import GraphVec

    db = GraphVec("mydb.db")
    db.add_node("alice", label="Person", name="Alice")

    beliefs = db.collection("beliefs")
    beliefs.add_node("b1", label="Belief", text="...")
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from graphvec.exceptions import CollectionNotFound
from graphvec.graph import Graph
from graphvec.storage.base import StorageBackend
from graphvec.storage.sqlite import SQLiteBackend

# Default collection name used when no explicit collection is chosen
_DEFAULT_COLLECTION = "default"


class GraphVec(Graph):
    """Embedded, serverless, persistent graph database with vector search.

    ``GraphVec`` is both the database handle **and** the default
    collection.  Call :py:meth:`collection` to open additional isolated
    graph namespaces within the same file.

    Args:
        path: Path to the ``.db`` file.  Use ``":memory:"`` for a
              transient in-memory database (tests / one-off scripts).
        embed_fn: Optional callable ``(text: str) -> list[float]`` that
                  produces embedding vectors.  When provided, nodes are
                  auto-embedded on insert if they contain *embed_field*.
        embed_field: The node property to pass to *embed_fn*
                     (default ``"text"``).
        backend: Advanced override — supply a custom
                 :class:`~graphvec.storage.base.StorageBackend`.  When
                 omitted, the built-in SQLite backend is used.

    Example::

        from graphvec import GraphVec

        db = GraphVec("kg.db")
        db.add_node("n1", label="Concept", name="graphvec")
        db.add_edge("n1", "n2", label="RELATED_TO")

        # Named collection (isolated namespace)
        docs = db.collection("documents")
        docs.add_node("d1", label="Document", title="…")
    """

    def __init__(
        self,
        path: str = ":memory:",
        *,
        embed_fn: Callable[[str], list[float]] | None = None,
        embed_field: str = "text",
        backend: StorageBackend | None = None,
    ) -> None:
        if backend is None:
            backend = SQLiteBackend(path)
        backend.open()
        self._db_backend = backend  # keep a ref for collection()
        self._embed_fn = embed_fn
        self._embed_field = embed_field
        super().__init__(
            backend=backend,
            collection=_DEFAULT_COLLECTION,
            embed_fn=embed_fn,
            embed_field=embed_field,
        )

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def collection(self, name: str) -> Graph:
        """Return (or create) a named collection.

        Each collection is an isolated graph namespace within the same
        database file.

        Args:
            name: Collection name.  Must be ASCII alphanumeric / underscores.

        Returns:
            A :class:`~graphvec.graph.Graph` scoped to *name*.
        """
        return Graph(
            backend=self._db_backend,
            collection=name,
            embed_fn=self._embed_fn,
            embed_field=self._embed_field,
        )

    def list_collections(self) -> list[str]:
        """Return the names of all collections in this database.

        Returns:
            Sorted list of collection name strings.
        """
        return self._db_backend.list_collections()

    def drop_collection(self, name: str) -> None:
        """Permanently delete a collection and all its data.

        Args:
            name: Collection to drop.

        Raises:
            CollectionNotFound: If the collection does not exist.
        """
        existing = self._db_backend.list_collections()
        if name not in existing:
            raise CollectionNotFound(name)
        self._db_backend.drop_collection(name)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the database connection.

        After calling this, the ``GraphVec`` instance should not be used.
        """
        self._db_backend.close()

    def __enter__(self) -> GraphVec:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"GraphVec(collections={self.list_collections()!r}, "
            f"nodes={self.node_count()}, edges={self.edge_count()})"
        )
