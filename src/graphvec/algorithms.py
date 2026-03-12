"""Pure-Python graph algorithms — no external dependencies required.

All algorithms operate on the abstract :class:`~graphvec.storage.base.StorageBackend`
interface, so they work with any storage implementation.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphvec.storage.base import StorageBackend


class Algorithms:
    """Graph algorithm implementations for a single collection.

    Args:
        backend: The storage backend providing graph data.
        collection: The collection namespace to operate within.
    """

    def __init__(self, backend: StorageBackend, collection: str) -> None:
        self._b = backend
        self._c = collection

    # ------------------------------------------------------------------
    # Degree
    # ------------------------------------------------------------------

    def degree(self, node_id: str) -> int:
        """Return the total degree (in + out) of *node_id*."""
        return self.in_degree(node_id) + self.out_degree(node_id)

    def in_degree(self, node_id: str) -> int:
        """Return the number of edges pointing **into** *node_id*."""
        return self._b.count_edges(self._c, dst=node_id)

    def out_degree(self, node_id: str) -> int:
        """Return the number of edges pointing **out of** *node_id*."""
        return self._b.count_edges(self._c, src=node_id)

    # ------------------------------------------------------------------
    # BFS / DFS
    # ------------------------------------------------------------------

    def bfs(self, start_id: str, max_depth: int = 10) -> list[str]:
        """Breadth-first traversal from *start_id*.

        Args:
            start_id: ID of the starting node.
            max_depth: Maximum number of hops to traverse.

        Returns:
            Ordered list of node IDs reached (including *start_id*).
        """
        visited: dict[str, int] = {start_id: 0}
        queue: deque[tuple[str, int]] = deque([(start_id, 0)])
        order: list[str] = []
        while queue:
            nid, depth = queue.popleft()
            order.append(nid)
            if depth >= max_depth:
                continue
            for edge in self._b.query_edges(self._c, src=nid):
                nbr = edge["dst"]
                if nbr not in visited:
                    visited[nbr] = depth + 1
                    queue.append((nbr, depth + 1))
        return order

    def dfs(self, start_id: str, max_depth: int = 10) -> list[str]:
        """Depth-first traversal from *start_id*.

        Args:
            start_id: ID of the starting node.
            max_depth: Maximum number of hops to traverse.

        Returns:
            Ordered list of node IDs reached (including *start_id*).
        """
        visited: set[str] = set()
        order: list[str] = []

        def _recurse(nid: str, depth: int) -> None:
            if nid in visited or depth > max_depth:
                return
            visited.add(nid)
            order.append(nid)
            for edge in self._b.query_edges(self._c, src=nid):
                _recurse(edge["dst"], depth + 1)

        _recurse(start_id, 0)
        return order

    # ------------------------------------------------------------------
    # Shortest path
    # ------------------------------------------------------------------

    def shortest_path(
        self,
        src: str,
        dst: str,
        max_hops: int = 20,
    ) -> list[str] | None:
        """BFS shortest path from *src* to *dst*.

        Args:
            src: Start node id.
            dst: Target node id.
            max_hops: Maximum path length to search.

        Returns:
            Ordered list of node IDs from *src* to *dst*, or ``None``
            if no path exists within *max_hops*.
        """
        if src == dst:
            return [src]
        parent: dict[str, str | None] = {src: None}
        queue: deque[tuple[str, int]] = deque([(src, 0)])
        while queue:
            nid, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for edge in self._b.query_edges(self._c, src=nid):
                nbr = edge["dst"]
                if nbr not in parent:
                    parent[nbr] = nid
                    if nbr == dst:
                        return _reconstruct(parent, src, dst)
                    queue.append((nbr, depth + 1))
        return None

    def all_paths(
        self,
        src: str,
        dst: str,
        max_hops: int = 6,
    ) -> list[list[str]]:
        """Return all simple paths from *src* to *dst* up to *max_hops* long.

        Args:
            src: Start node id.
            dst: Target node id.
            max_hops: Maximum path length.

        Returns:
            List of paths, each a list of node IDs.
        """
        results: list[list[str]] = []
        self._dfs_paths(src, dst, max_hops, [src], set(), results)
        return results

    def _dfs_paths(
        self,
        current: str,
        dst: str,
        max_hops: int,
        path: list[str],
        visited: set[str],
        results: list[list[str]],
    ) -> None:
        if current == dst and len(path) > 1:
            results.append(list(path))
            return
        if len(path) - 1 >= max_hops:
            return
        visited.add(current)
        for edge in self._b.query_edges(self._c, src=current):
            nbr = edge["dst"]
            if nbr not in visited:
                path.append(nbr)
                self._dfs_paths(nbr, dst, max_hops, path, visited, results)
                path.pop()
        visited.discard(current)

    def neighbors(self, node_id: str, hops: int = 1) -> set[str]:
        """Return all node IDs reachable from *node_id* within *hops* steps.

        Args:
            node_id: The starting node.
            hops: Maximum number of hops (default ``1``).

        Returns:
            Set of reachable node IDs (excluding *node_id* itself).
        """
        reached: set[str] = set()
        frontier: set[str] = {node_id}
        for _ in range(hops):
            next_frontier: set[str] = set()
            for nid in frontier:
                for edge in self._b.query_edges(self._c, src=nid):
                    nbr = edge["dst"]
                    if nbr != node_id and nbr not in reached:
                        next_frontier.add(nbr)
            reached |= next_frontier
            frontier = next_frontier
            if not frontier:
                break
        return reached

    # ------------------------------------------------------------------
    # Connected components
    # ------------------------------------------------------------------

    def connected_components(self) -> list[set[str]]:
        """Return all weakly connected components.

        Fetches all edges once and builds an in-memory adjacency structure
        to avoid N×2 round-trips for large graphs.

        Returns:
            List of sets of node IDs, one set per component.
        """
        all_nodes = {r["id"] for r in self._b.query_nodes(self._c)}
        if not all_nodes:
            return []

        # Build undirected adjacency from a single edge query
        adj: dict[str, list[str]] = {nid: [] for nid in all_nodes}
        for e in self._b.query_edges(self._c):
            src, dst = e["src"], e["dst"]
            if src in adj:
                adj[src].append(dst)
            if dst in adj:
                adj[dst].append(src)

        visited: set[str] = set()
        components: list[set[str]] = []

        def _component(start: str) -> set[str]:
            comp: set[str] = set()
            queue: deque[str] = deque([start])
            while queue:
                nid = queue.popleft()
                if nid in comp:
                    continue
                comp.add(nid)
                for nbr in adj.get(nid, []):
                    if nbr not in comp:
                        queue.append(nbr)
            return comp

        for nid in all_nodes:
            if nid not in visited:
                comp = _component(nid)
                components.append(comp)
                visited |= comp

        return components

    def is_connected(self) -> bool:
        """Return ``True`` if the graph is weakly connected."""
        comps = self.connected_components()
        return len(comps) <= 1

    # ------------------------------------------------------------------
    # PageRank
    # ------------------------------------------------------------------

    def pagerank(
        self,
        damping: float = 0.85,
        iterations: int = 50,
        tol: float = 1e-6,
    ) -> dict[str, float]:
        """Compute PageRank scores for all nodes.

        Args:
            damping: Damping factor (probability of following an edge).
            iterations: Maximum number of power-iteration steps.
            tol: Convergence threshold (L1 norm of score delta).

        Returns:
            ``{node_id: score}`` mapping.  Scores sum to approximately
            ``1.0``.
        """
        node_ids = [r["id"] for r in self._b.query_nodes(self._c)]
        n = len(node_ids)
        if n == 0:
            return {}
        idx: dict[str, int] = {nid: i for i, nid in enumerate(node_ids)}
        scores = [1.0 / n] * n

        # Build adjacency lists
        out_neighbours: list[list[int]] = [[] for _ in range(n)]
        for edge in self._b.query_edges(self._c):
            si, di = idx.get(edge["src"]), idx.get(edge["dst"])
            if si is not None and di is not None:
                out_neighbours[si].append(di)

        out_degree_arr = [len(out_neighbours[i]) for i in range(n)]

        for _ in range(iterations):
            new_scores = [(1.0 - damping) / n] * n
            for i in range(n):
                if out_degree_arr[i] == 0:
                    # Dangling node — distribute to all
                    share = damping * scores[i] / n
                    for j in range(n):
                        new_scores[j] += share
                else:
                    share = damping * scores[i] / out_degree_arr[i]
                    for j in out_neighbours[i]:
                        new_scores[j] += share
            delta = sum(abs(new_scores[i] - scores[i]) for i in range(n))
            scores = new_scores
            if delta < tol:
                break

        return {node_ids[i]: scores[i] for i in range(n)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reconstruct(parent: dict[str, str | None], src: str, dst: str) -> list[str]:
    path: list[str] = []
    current: str | None = dst
    while current is not None:
        path.append(current)
        current = parent.get(current)
    path.reverse()
    return path
