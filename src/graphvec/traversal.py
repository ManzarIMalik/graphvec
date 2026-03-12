"""Fluent, chainable graph traversal API.

The :class:`Traversal` object is the return type of ``g.v()``.
It accumulates traversal steps lazily and executes them only when a
terminal method (``.all()``, ``.first()``, ``.count()``, …) is called.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from graphvec.models import Node, SearchResult

if TYPE_CHECKING:
    from graphvec.storage.base import StorageBackend


def _props_match(row: dict[str, Any], kwargs: dict[str, Any]) -> bool:
    """Return True if all kwargs match the node row's properties."""
    props = row["properties"]
    return all(props.get(k) == v for k, v in kwargs.items())


class Traversal:
    """A lazy, chainable graph traversal builder.

    Instances are created by :py:meth:`graphvec.graph.Graph.v` and
    (for vector-hybrid queries) by the ``SearchResultSet`` wrapper.
    Do not construct directly.
    """

    def __init__(
        self,
        backend: StorageBackend,
        collection: str,
        seed_nodes: list[dict[str, Any]],
    ) -> None:
        self._backend = backend
        self._collection = collection
        # Current working set of node rows
        self._nodes: list[dict[str, Any]] = seed_nodes
        self._limit_val: int | None = None
        self._skip_val: int = 0

    # ------------------------------------------------------------------
    # Traversal steps
    # ------------------------------------------------------------------

    def out(self, label: str | None = None, hops: int = 1) -> Traversal:
        """Follow outgoing edges from the current node set.

        Args:
            label: If given, only follow edges with this label.
            hops: Number of times to repeat the step (default ``1``).
        """
        for _ in range(hops):
            next_nodes: dict[str, dict[str, Any]] = {}
            for node in self._nodes:
                edges = self._backend.query_edges(
                    self._collection, label=label, src=node["id"]
                )
                for edge in edges:
                    nbr = self._backend.fetch_node(self._collection, edge["dst"])
                    if nbr and nbr["id"] not in next_nodes:
                        next_nodes[nbr["id"]] = nbr
            self._nodes = list(next_nodes.values())
        return self

    def in_(self, label: str | None = None, hops: int = 1) -> Traversal:
        """Follow incoming edges into the current node set.

        Args:
            label: If given, only follow edges with this label.
            hops: Number of times to repeat the step (default ``1``).
        """
        for _ in range(hops):
            next_nodes: dict[str, dict[str, Any]] = {}
            for node in self._nodes:
                edges = self._backend.query_edges(
                    self._collection, label=label, dst=node["id"]
                )
                for edge in edges:
                    nbr = self._backend.fetch_node(self._collection, edge["src"])
                    if nbr and nbr["id"] not in next_nodes:
                        next_nodes[nbr["id"]] = nbr
            self._nodes = list(next_nodes.values())
        return self

    def both(self, label: str | None = None, hops: int = 1) -> Traversal:
        """Follow edges in **either** direction from the current node set.

        Args:
            label: If given, only follow edges with this label.
            hops: Number of times to repeat the step (default ``1``).
        """
        for _ in range(hops):
            next_nodes: dict[str, dict[str, Any]] = {}
            for node in self._nodes:
                for direction in ("src", "dst"):
                    kwargs: dict[str, Any] = {"label": label}
                    if direction == "src":
                        kwargs["src"] = node["id"]
                        other_key = "dst"
                    else:
                        kwargs["dst"] = node["id"]
                        other_key = "src"
                    for edge in self._backend.query_edges(self._collection, **kwargs):
                        nbr = self._backend.fetch_node(
                            self._collection, edge[other_key]
                        )
                        if nbr and nbr["id"] not in next_nodes:
                            next_nodes[nbr["id"]] = nbr
            self._nodes = list(next_nodes.values())
        return self

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def has(self, **kwargs: Any) -> Traversal:
        """Keep only nodes whose properties match all *kwargs* equality checks.

        Example::

            g.v(label="Person").has(active=True)
        """
        self._nodes = [n for n in self._nodes if _props_match(n, kwargs)]
        return self

    def has_label(self, label: str) -> Traversal:
        """Keep only nodes whose label matches *label*."""
        self._nodes = [n for n in self._nodes if n.get("label") == label]
        return self

    def has_not(self, **kwargs: Any) -> Traversal:
        """Keep only nodes that do **not** match all *kwargs* conditions.

        Example::

            g.v().has_not(archived=True)
        """
        self._nodes = [n for n in self._nodes if not _props_match(n, kwargs)]
        return self

    def where(self, fn: Callable[[Node], bool]) -> Traversal:
        """Keep only nodes for which the predicate *fn* returns ``True``.

        Example::

            g.v(label="Claim").where(lambda n: n["confidence"] > 0.8)
        """
        self._nodes = [n for n in self._nodes if fn(Node.from_row(n))]
        return self

    # ------------------------------------------------------------------
    # Pagination helpers (applied at terminal time)
    # ------------------------------------------------------------------

    def limit(self, n: int) -> Traversal:
        """Cap the result set to at most *n* items."""
        self._limit_val = n
        return self

    def skip(self, n: int) -> Traversal:
        """Skip the first *n* items (offset for pagination)."""
        self._skip_val = n
        return self

    # ------------------------------------------------------------------
    # Terminal operations
    # ------------------------------------------------------------------

    def _finalise(self) -> list[dict[str, Any]]:
        nodes = self._nodes
        if self._skip_val:
            nodes = nodes[self._skip_val:]
        if self._limit_val is not None:
            nodes = nodes[: self._limit_val]
        return nodes

    def all(self) -> list[Node]:
        """Return all nodes in the current traversal as :class:`~graphvec.models.Node` objects."""
        return [Node.from_row(n) for n in self._finalise()]

    def first(self) -> Node | None:
        """Return the first node, or ``None`` if the traversal is empty."""
        nodes = self._finalise()
        return Node.from_row(nodes[0]) if nodes else None

    def count(self) -> int:
        """Return the number of nodes in the current traversal."""
        return len(self._finalise())

    def ids(self) -> list[str]:
        """Return a list of node IDs in the current traversal."""
        return [n["id"] for n in self._finalise()]

    def to_dataframe(self) -> Any:
        """Return a ``pandas.DataFrame`` of all nodes.

        Requires ``pandas`` (install with ``graphvec[pandas]``).

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "pandas is required for .to_dataframe(). "
                "Install with: pip install graphvec[pandas]"
            ) from exc
        records = []
        for n in self._finalise():
            row: dict[str, Any] = {
                "id": n["id"],
                "label": n.get("label", ""),
                "created_at": n.get("created_at", 0.0),
                "updated_at": n.get("updated_at", 0.0),
            }
            row.update(n.get("properties", {}))
            records.append(row)
        return pd.DataFrame(records)


class SearchTraversal:
    """A traversal seeded from vector search results.

    Wraps a list of :class:`~graphvec.models.SearchResult` objects and
    exposes the same fluent traversal API so that search results can be
    fed directly into graph traversal steps::

        g.search(vec, k=5).out("RELATED_TO").all()
    """

    def __init__(
        self,
        results: list[SearchResult],
        backend: StorageBackend,
        collection: str,
    ) -> None:
        self._results = results
        self._backend = backend
        self._collection = collection

    # Delegate graph traversal steps to an inner Traversal
    def _as_traversal(self) -> Traversal:
        seed = []
        for r in self._results:
            row = self._backend.fetch_node(self._collection, r.node.id)
            if row:
                seed.append(row)
        return Traversal(self._backend, self._collection, seed)

    def out(self, label: str | None = None, hops: int = 1) -> Traversal:
        """Follow outgoing edges from search result nodes."""
        return self._as_traversal().out(label, hops)

    def in_(self, label: str | None = None, hops: int = 1) -> Traversal:
        """Follow incoming edges into search result nodes."""
        return self._as_traversal().in_(label, hops)

    def both(self, label: str | None = None, hops: int = 1) -> Traversal:
        """Follow edges in either direction from search result nodes."""
        return self._as_traversal().both(label, hops)

    def has(self, **kwargs: Any) -> Traversal:
        """Filter search result nodes by property equality."""
        return self._as_traversal().has(**kwargs)

    def has_label(self, label: str) -> Traversal:
        """Filter search result nodes by label."""
        return self._as_traversal().has_label(label)

    def where(self, fn: Callable[[Node], bool]) -> Traversal:
        """Filter search result nodes with a predicate."""
        return self._as_traversal().where(fn)

    # Iterator support — iterate over SearchResult objects
    def __iter__(self):  # type: ignore[override]
        return iter(self._results)

    def __len__(self) -> int:
        return len(self._results)

    def __getitem__(self, idx: int) -> SearchResult:
        return self._results[idx]
