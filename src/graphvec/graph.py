"""Core Graph class — the primary interface for node/edge CRUD,
traversal, vector search, algorithms, and transactions.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

from graphvec.algorithms import Algorithms
from graphvec.exceptions import EdgeNotFound, NodeNotFound, StorageError
from graphvec.index import IndexManager
from graphvec.io import IO
from graphvec.models import Edge, Node, Path
from graphvec.storage.base import StorageBackend
from graphvec.transaction import Transaction
from graphvec.traversal import SearchTraversal, Traversal
from graphvec.vector import VectorStore


class Graph:
    """A property graph backed by a persistent storage backend.

    Do not construct directly — use :class:`~graphvec.db.GraphVec` or
    :py:meth:`~graphvec.db.GraphVec.collection`.

    Args:
        backend: Storage backend instance (already opened).
        collection: Collection namespace for all operations.
        embed_fn: Optional callable ``(text: str) -> list[float]`` used
                  for automatic embedding generation on node insert.
        embed_field: Node property whose value is passed to *embed_fn*.
    """

    def __init__(
        self,
        backend: StorageBackend,
        collection: str,
        embed_fn: Callable[[str], list[float]] | None = None,
        embed_field: str = "text",
    ) -> None:
        self._backend = backend
        self._collection = collection
        self._embed_fn = embed_fn
        self._embed_field = embed_field
        self._vectors = VectorStore(backend, collection)
        self._algo = Algorithms(backend, collection)
        self._indexes = IndexManager(backend, collection)
        self._io = IO(self)

    # ------------------------------------------------------------------
    # Node CRUD
    # ------------------------------------------------------------------

    def add_node(self, id: str, label: str = "", **properties: Any) -> Node:
        """Add a node to the graph.

        Args:
            id: Unique node identifier.
            label: Semantic label for the node (e.g. ``"Person"``).
            **properties: Arbitrary key-value metadata.

        Returns:
            The newly created :class:`~graphvec.models.Node`.
        """
        now = time.time()
        row: dict[str, Any] = {
            "id": id,
            "label": label,
            "properties": properties,
            "created_at": now,
            "updated_at": now,
        }
        self._backend.insert_node(self._collection, row)
        # Auto-embed if configured
        if self._embed_fn and self._embed_field in properties:
            text = properties[self._embed_field]
            if isinstance(text, str):
                vector = self._embed_fn(text)
                self._vectors.set_embedding(id, vector)
        self._backend.autocommit()
        return Node.from_row(row)

    def get_node(self, id: str) -> Node | None:
        """Return the node with *id*, or ``None`` if it does not exist.

        Args:
            id: Node identifier.
        """
        row = self._backend.fetch_node(self._collection, id)
        return Node.from_row(row) if row else None

    def update_node(self, id: str, **properties: Any) -> Node:
        """Merge *properties* into an existing node's property dict.

        Args:
            id: Node identifier.
            **properties: Properties to update or add.

        Returns:
            The updated :class:`~graphvec.models.Node`.

        Raises:
            NodeNotFound: If the node does not exist.
        """
        if not self.node_exists(id):
            raise NodeNotFound(id)
        self._backend.update_node(
            self._collection, id, {"properties": properties, "updated_at": time.time()}
        )
        self._backend.autocommit()
        return self.get_node(id)  # type: ignore[return-value]

    def delete_node(self, id: str) -> None:
        """Delete a node and all its connected edges / embeddings.

        Args:
            id: Node identifier.

        Raises:
            NodeNotFound: If the node does not exist.
        """
        if not self.node_exists(id):
            raise NodeNotFound(id)
        self._backend.delete_node(self._collection, id)
        self._backend.autocommit()

    def nodes(self, label: str | None = None, **filters: Any) -> list[Node]:
        """Return nodes, optionally filtered.

        Args:
            label: If given, only return nodes with this label.
            **filters: Property equality filters.

        Returns:
            List of :class:`~graphvec.models.Node` objects.
        """
        rows = self._backend.query_nodes(
            self._collection,
            label=label,
            filters=filters if filters else None,
        )
        return [Node.from_row(r) for r in rows]

    def node_count(self) -> int:
        """Return the total number of nodes in the graph."""
        return len(self._backend.query_nodes(self._collection))

    def node_exists(self, id: str) -> bool:
        """Return ``True`` if a node with *id* exists.

        Args:
            id: Node identifier.
        """
        return self._backend.fetch_node(self._collection, id) is not None

    def add_nodes(self, node_list: list[dict[str, Any]]) -> list[Node]:
        """Bulk insert multiple nodes in a single transaction.

        Args:
            node_list: List of dicts, each with at least ``"id"`` and
                       optionally ``"label"`` and property key-value pairs.

        Returns:
            List of created :class:`~graphvec.models.Node` objects.
        """
        results: list[Node] = []
        with self.transaction():
            for data in node_list:
                data = dict(data)
                nid = data.pop("id")
                label = data.pop("label", "")
                results.append(self.add_node(nid, label, **data))
        return results

    # ------------------------------------------------------------------
    # Edge CRUD
    # ------------------------------------------------------------------

    def add_edge(
        self,
        src: str,
        dst: str,
        label: str = "",
        weight: float = 1.0,
        **properties: Any,
    ) -> Edge:
        """Add a directed edge from *src* to *dst*.

        Args:
            src: Source node id.
            dst: Destination node id.
            label: Relationship type (e.g. ``"KNOWS"``).
            weight: Numeric edge weight (default ``1.0``).
            **properties: Arbitrary key-value metadata.

        Returns:
            The newly created :class:`~graphvec.models.Edge`.

        Raises:
            NodeNotFound: If either *src* or *dst* does not exist.
        """
        if not self.node_exists(src):
            raise NodeNotFound(src)
        if not self.node_exists(dst):
            raise NodeNotFound(dst)
        now = time.time()
        edge_id = str(uuid.uuid4())
        row: dict[str, Any] = {
            "id": edge_id,
            "src": src,
            "dst": dst,
            "label": label,
            "properties": properties,
            "weight": weight,
            "created_at": now,
            "updated_at": now,
        }
        self._backend.insert_edge(self._collection, row)
        self._backend.autocommit()
        return Edge.from_row(row)

    def get_edge(self, id: str) -> Edge | None:
        """Return the edge with *id*, or ``None`` if not found.

        Args:
            id: Edge identifier.
        """
        row = self._backend.fetch_edge(self._collection, id)
        return Edge.from_row(row) if row else None

    def update_edge(self, id: str, **properties: Any) -> Edge:
        """Merge *properties* into an existing edge.

        Args:
            id: Edge identifier.
            **properties: Properties to update or add.

        Returns:
            The updated :class:`~graphvec.models.Edge`.

        Raises:
            EdgeNotFound: If the edge does not exist.
        """
        row = self._backend.fetch_edge(self._collection, id)
        if row is None:
            raise EdgeNotFound(id)
        self._backend.update_edge(
            self._collection, id, {"properties": properties, "updated_at": time.time()}
        )
        self._backend.autocommit()
        return self.get_edge(id)  # type: ignore[return-value]

    def delete_edge(self, id: str) -> None:
        """Delete an edge by id.

        Args:
            id: Edge identifier.

        Raises:
            EdgeNotFound: If the edge does not exist.
        """
        if self._backend.fetch_edge(self._collection, id) is None:
            raise EdgeNotFound(id)
        self._backend.delete_edge(self._collection, id)
        self._backend.autocommit()

    def edges(
        self,
        label: str | None = None,
        src: str | None = None,
        dst: str | None = None,
        **filters: Any,
    ) -> list[Edge]:
        """Return edges, optionally filtered.

        Args:
            label: If given, only return edges with this label.
            src: If given, only return edges from this node.
            dst: If given, only return edges to this node.
            **filters: Property equality filters.

        Returns:
            List of :class:`~graphvec.models.Edge` objects.
        """
        rows = self._backend.query_edges(
            self._collection,
            label=label,
            src=src,
            dst=dst,
            filters=filters if filters else None,
        )
        return [Edge.from_row(r) for r in rows]

    def edge_count(self) -> int:
        """Return the total number of edges in the graph."""
        return len(self._backend.query_edges(self._collection))

    def edge_exists(self, src: str, dst: str, label: str | None = None) -> bool:
        """Return ``True`` if at least one edge from *src* to *dst* exists.

        Args:
            src: Source node id.
            dst: Destination node id.
            label: If given, also match on edge label.
        """
        rows = self._backend.query_edges(
            self._collection, label=label, src=src, dst=dst
        )
        return len(rows) > 0

    def add_edges(self, edge_list: list[dict[str, Any]]) -> list[Edge]:
        """Bulk insert multiple edges in a single transaction.

        Args:
            edge_list: List of dicts with at least ``"src"``, ``"dst"``,
                       and optionally ``"label"``, ``"weight"``, and
                       property key-value pairs.

        Returns:
            List of created :class:`~graphvec.models.Edge` objects.
        """
        results: list[Edge] = []
        with self.transaction():
            for data in edge_list:
                data = dict(data)
                src = data.pop("src")
                dst = data.pop("dst")
                label = data.pop("label", "")
                weight = data.pop("weight", 1.0)
                results.append(self.add_edge(src, dst, label, weight=weight, **data))
        return results

    # ------------------------------------------------------------------
    # Traversal API
    # ------------------------------------------------------------------

    def v(
        self,
        node_id: str | None = None,
        label: str | None = None,
    ) -> Traversal:
        """Start a fluent traversal.

        Args:
            node_id: If given, start at this specific node.
            label: If given, start at all nodes with this label.

        Returns:
            A :class:`~graphvec.traversal.Traversal` builder.

        Example::

            g.v("n1").out("KNOWS").has(active=True).all()
        """
        if node_id is not None:
            row = self._backend.fetch_node(self._collection, node_id)
            seed = [row] if row else []
        elif label is not None:
            seed = self._backend.query_nodes(self._collection, label=label)
        else:
            seed = self._backend.query_nodes(self._collection)
        return Traversal(self._backend, self._collection, seed)

    # ------------------------------------------------------------------
    # Path finding
    # ------------------------------------------------------------------

    def path(
        self, src: str, dst: str, max_hops: int = 20
    ) -> Path | None:
        """Find the shortest path from *src* to *dst*.

        Args:
            src: Start node id.
            dst: Target node id.
            max_hops: Maximum path length.

        Returns:
            A :class:`~graphvec.models.Path`, or ``None`` if unreachable.
        """
        node_ids = self._algo.shortest_path(src, dst, max_hops)
        if node_ids is None:
            return None
        return self._ids_to_path(node_ids)

    def all_paths(
        self, src: str, dst: str, max_hops: int = 6
    ) -> list[Path]:
        """Return all simple paths from *src* to *dst*.

        Args:
            src: Start node id.
            dst: Target node id.
            max_hops: Maximum path length.

        Returns:
            List of :class:`~graphvec.models.Path` objects.
        """
        paths_ids = self._algo.all_paths(src, dst, max_hops)
        return [self._ids_to_path(ids) for ids in paths_ids]

    def neighbors(self, node_id: str, hops: int = 1) -> list[Node]:
        """Return all nodes reachable from *node_id* within *hops* steps.

        Args:
            node_id: The starting node.
            hops: Maximum number of hops.

        Returns:
            List of :class:`~graphvec.models.Node` objects.
        """
        ids = self._algo.neighbors(node_id, hops)
        rows = self._backend.fetch_nodes_by_ids(self._collection, list(ids))
        return [Node.from_row(r) for r in rows]

    def _ids_to_path(self, node_ids: list[str]) -> Path:
        # Bulk-fetch all path nodes in one query, then re-order by path sequence.
        row_map = {
            r["id"]: r
            for r in self._backend.fetch_nodes_by_ids(self._collection, node_ids)
        }
        nodes = [Node.from_row(row_map[nid]) for nid in node_ids if nid in row_map]
        edges: list[Edge] = []
        for i in range(len(node_ids) - 1):
            edge_rows = self._backend.query_edges(
                self._collection, src=node_ids[i], dst=node_ids[i + 1]
            )
            if edge_rows:
                edges.append(Edge.from_row(edge_rows[0]))
        return Path(nodes=nodes, edges=edges, length=len(edges))

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    def set_embedding(
        self,
        node_id: str,
        vector: list[float],
        model: str = "custom",
    ) -> None:
        """Store (or replace) a vector embedding for *node_id*.

        Args:
            node_id: Target node id.
            vector: Float embedding vector.
            model: Human-readable model label.

        Raises:
            NodeNotFound: If *node_id* does not exist.
        """
        if not self.node_exists(node_id):
            raise NodeNotFound(node_id)
        self._vectors.set_embedding(node_id, vector, model)
        self._backend.autocommit()

    def get_embedding(self, node_id: str) -> list[float]:
        """Return the embedding vector for *node_id*.

        Raises:
            EmbeddingNotFound: If no embedding is stored.
        """
        return self._vectors.get_embedding(node_id)

    def search(
        self,
        query_vector: list[float],
        k: int = 5,
        metric: str = "cosine",
        label: str | None = None,
    ) -> SearchTraversal:
        """Vector similarity search.

        Finds the *k* nodes whose stored embeddings are most similar to
        *query_vector*, then returns a :class:`~graphvec.traversal.SearchTraversal`
        that can be used directly or chained into further graph traversal.

        Args:
            query_vector: Query embedding.
            k: Number of results to return.
            metric: ``"cosine"`` (default), ``"euclidean"``, or ``"dot"``.
            label: If given, restrict search to nodes with this label.

        Returns:
            A :class:`~graphvec.traversal.SearchTraversal` (iterable +
            graph-traversal capable).

        Example::

            results = g.search(vec, k=5)
            for r in results:
                print(r.node.id, r.score)

            # Hybrid: search then traverse
            g.search(vec, k=3).out("RELATED_TO").all()
        """
        node_rows = self._backend.query_nodes(self._collection)
        results = self._vectors.search(
            query_vector,
            k=k,
            metric=metric,
            label_filter=label,
            node_rows=node_rows,
        )
        return SearchTraversal(results, self._backend, self._collection)

    def search_text(
        self,
        text: str,
        k: int = 5,
        metric: str = "cosine",
        label: str | None = None,
    ) -> SearchTraversal:
        """Embed *text* with the configured ``embed_fn`` then call :meth:`search`.

        Args:
            text: The query string to embed.
            k: Number of results.
            metric: Distance metric.
            label: Optional node label filter.

        Raises:
            StorageError: If no ``embed_fn`` was configured.
        """
        if self._embed_fn is None:
            raise StorageError(
                "search_text() requires an embed_fn. "
                "Pass one when constructing GraphVec: "
                "GraphVec('db.db', embed_fn=my_fn)"
            )
        vector = self._embed_fn(text)
        return self.search(vector, k=k, metric=metric, label=label)

    # ------------------------------------------------------------------
    # Graph algorithms — delegated to Algorithms
    # ------------------------------------------------------------------

    def degree(self, node_id: str) -> int:
        """Total degree of *node_id* (in + out)."""
        return self._algo.degree(node_id)

    def in_degree(self, node_id: str) -> int:
        """Number of edges pointing into *node_id*."""
        return self._algo.in_degree(node_id)

    def out_degree(self, node_id: str) -> int:
        """Number of edges pointing out of *node_id*."""
        return self._algo.out_degree(node_id)

    def bfs(self, start_id: str, max_depth: int = 10) -> list[str]:
        """Breadth-first traversal from *start_id*.

        Returns:
            Ordered list of node IDs visited.
        """
        return self._algo.bfs(start_id, max_depth)

    def dfs(self, start_id: str, max_depth: int = 10) -> list[str]:
        """Depth-first traversal from *start_id*.

        Returns:
            Ordered list of node IDs visited.
        """
        return self._algo.dfs(start_id, max_depth)

    def shortest_path(self, src: str, dst: str, max_hops: int = 20) -> list[str] | None:
        """Return the shortest path as a list of node IDs, or ``None``."""
        return self._algo.shortest_path(src, dst, max_hops)

    def connected_components(self) -> list[set[str]]:
        """Return all weakly connected components as sets of node IDs."""
        return self._algo.connected_components()

    def is_connected(self) -> bool:
        """Return ``True`` if the entire graph is weakly connected."""
        return self._algo.is_connected()

    def pagerank(
        self, damping: float = 0.85, iterations: int = 50
    ) -> dict[str, float]:
        """Compute PageRank scores.

        Returns:
            ``{node_id: score}`` mapping.
        """
        return self._algo.pagerank(damping, iterations)

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------

    @contextmanager
    def transaction(self) -> Generator[Transaction, None, None]:
        """Context-manager transaction.

        All graph mutations within the ``with`` block are committed
        atomically; any exception triggers a full rollback.

        Example::

            with g.transaction():
                g.add_node("n1", label="Claim")
                g.add_edge("n1", "n2", label="SUPPORTS")
        """
        txn = Transaction(self._backend)
        txn.begin()
        try:
            yield txn
            txn.commit()
        except Exception:
            txn.rollback()
            raise

    def begin(self) -> Transaction:
        """Start a manual transaction.

        Returns:
            An open :class:`~graphvec.transaction.Transaction`.
        """
        return Transaction(self._backend).begin()

    # ------------------------------------------------------------------
    # Index management — delegated to IndexManager
    # ------------------------------------------------------------------

    def create_index(self, target: str, field: str) -> None:
        """Create an index.  See :class:`~graphvec.index.IndexManager`."""
        self._indexes.create_index(target, field)

    def drop_index(self, target: str, field: str) -> None:
        """Drop an index.  See :class:`~graphvec.index.IndexManager`."""
        self._indexes.drop_index(target, field)

    def list_indexes(self) -> list[dict[str, Any]]:
        """List all indexes for this collection."""
        return self._indexes.list_indexes()

    # ------------------------------------------------------------------
    # Import / export — delegated to IO
    # ------------------------------------------------------------------

    def export_json(self, path: str) -> None:
        """Export the full graph to a JSON file."""
        self._io.export_json(path)

    def import_json(self, path: str) -> None:
        """Import (merge) a graph from a JSON snapshot."""
        self._io.import_json(path)

    def export_csv(self, nodes_path: str, edges_path: str) -> None:
        """Export nodes and edges to two CSV files."""
        self._io.export_csv(nodes_path, edges_path)

    def import_csv(self, nodes_path: str, edges_path: str) -> None:
        """Import nodes and edges from two CSV files."""
        self._io.import_csv(nodes_path, edges_path)

    def to_networkx(self) -> Any:
        """Return a ``networkx.DiGraph`` representation of the graph."""
        return self._io.to_networkx()

    def from_networkx(self, nx_graph: Any) -> None:
        """Ingest a NetworkX graph."""
        self._io.from_networkx(nx_graph)

    # ------------------------------------------------------------------
    # Subgraph
    # ------------------------------------------------------------------

    def subgraph(self, node_ids: list[str]) -> Subgraph:
        """Return a read-only subgraph view over *node_ids*.

        Args:
            node_ids: Node IDs to include.

        Returns:
            A :class:`Subgraph` that exposes ``export_json``,
            ``visualize``, and ``nodes`` / ``edges`` scoped to the
            given IDs.
        """
        return Subgraph(self, node_ids)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize(
        self,
        output: str | None = None,
        highlight: list[str] | None = None,
    ) -> None:
        """Render the graph.

        Args:
            output: File path to save the image, or ``None`` for a
                    live window.
            highlight: Node IDs to render with a contrasting colour.

        Requires ``graphvec[viz]``.
        """
        from graphvec.viz import visualize as _viz

        _viz(self, output=output, highlight=highlight)


class Subgraph:
    """A lightweight view over a subset of graph nodes.

    Created via :py:meth:`Graph.subgraph`.
    """

    def __init__(self, graph: Graph, node_ids: list[str]) -> None:
        self._graph = graph
        self._node_ids: set[str] = set(node_ids)

    def nodes(self) -> list[Node]:
        """Return the nodes in this subgraph."""
        return [n for n in self._graph.nodes() if n.id in self._node_ids]

    def edges(self) -> list[Edge]:
        """Return edges where both endpoints are in this subgraph."""
        return [
            e for e in self._graph.edges()
            if e.src in self._node_ids and e.dst in self._node_ids
        ]

    def export_json(self, path: str) -> None:
        """Export this subgraph to a JSON file.

        Args:
            path: Destination file path.
        """
        import json
        import time as _time

        snapshot = {
            "version": 1,
            "exported_at": _time.time(),
            "nodes": [
                {
                    "id": n.id,
                    "label": n.label,
                    "properties": n.properties,
                    "created_at": n.created_at,
                    "updated_at": n.updated_at,
                }
                for n in self.nodes()
            ],
            "edges": [
                {
                    "id": e.id,
                    "src": e.src,
                    "dst": e.dst,
                    "label": e.label,
                    "properties": e.properties,
                    "weight": e.weight,
                    "created_at": e.created_at,
                    "updated_at": e.updated_at,
                }
                for e in self.edges()
            ],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(snapshot, fh, indent=2)

    def visualize(
        self,
        output: str | None = None,
        highlight: list[str] | None = None,
    ) -> None:
        """Visualise this subgraph.

        Requires ``graphvec[viz]``.
        """
        from graphvec.viz import visualize as _viz

        _viz(
            self._graph,  # type: ignore[arg-type]
            output=output,
            highlight=highlight,
            node_ids=list(self._node_ids),
        )
