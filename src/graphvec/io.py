"""Import / export utilities for graphvec graphs.

Supports JSON snapshots, CSV files, and NetworkX interoperability.
All operations go through the :class:`~graphvec.graph.Graph` public API
so they work with any storage backend.
"""

from __future__ import annotations

import csv
import json
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from graphvec.graph import Graph


class IO:
    """Import/export helper bound to a :class:`~graphvec.graph.Graph`.

    Args:
        graph: The graph instance to read from / write to.
    """

    def __init__(self, graph: Graph) -> None:
        self._g = graph

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def export_json(self, path: str) -> None:
        """Write a full graph snapshot to a JSON file.

        The snapshot includes all nodes and edges with their properties.

        Args:
            path: Destination file path.
        """
        snapshot: dict[str, Any] = {
            "version": 1,
            "exported_at": time.time(),
            "nodes": [],
            "edges": [],
        }
        for node in self._g.nodes():
            snapshot["nodes"].append(
                {
                    "id": node.id,
                    "label": node.label,
                    "properties": node.properties,
                    "created_at": node.created_at,
                    "updated_at": node.updated_at,
                }
            )
        for edge in self._g.edges():
            snapshot["edges"].append(
                {
                    "id": edge.id,
                    "src": edge.src,
                    "dst": edge.dst,
                    "label": edge.label,
                    "properties": edge.properties,
                    "weight": edge.weight,
                    "created_at": edge.created_at,
                    "updated_at": edge.updated_at,
                }
            )
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(snapshot, fh, indent=2)

    def import_json(self, path: str) -> None:
        """Load a graph snapshot from a JSON file (merges with existing data).

        Existing nodes/edges with the same ID are skipped.

        Args:
            path: Source file path.
        """
        with open(path, encoding="utf-8") as fh:
            snapshot: dict[str, Any] = json.load(fh)

        for node_data in snapshot.get("nodes", []):
            nid = node_data["id"]
            if not self._g.node_exists(nid):
                props = dict(node_data.get("properties", {}))
                self._g.add_node(nid, node_data.get("label", ""), **props)

        for edge_data in snapshot.get("edges", []):
            props = dict(edge_data.get("properties", {}))
            self._g.add_edge(
                edge_data["src"],
                edge_data["dst"],
                edge_data.get("label", ""),
                weight=edge_data.get("weight", 1.0),
                **props,
            )

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def export_csv(self, nodes_path: str, edges_path: str) -> None:
        """Export graph to two CSV files (one for nodes, one for edges).

        Properties are JSON-encoded in a single column.

        Args:
            nodes_path: Path for the nodes CSV.
            edges_path: Path for the edges CSV.
        """
        with open(nodes_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["id", "label", "properties", "created_at", "updated_at"])
            for node in self._g.nodes():
                writer.writerow(
                    [
                        node.id,
                        node.label,
                        json.dumps(node.properties),
                        node.created_at,
                        node.updated_at,
                    ]
                )

        with open(edges_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["id", "src", "dst", "label", "properties", "weight",
                 "created_at", "updated_at"]
            )
            for edge in self._g.edges():
                writer.writerow(
                    [
                        edge.id,
                        edge.src,
                        edge.dst,
                        edge.label,
                        json.dumps(edge.properties),
                        edge.weight,
                        edge.created_at,
                        edge.updated_at,
                    ]
                )

    def import_csv(self, nodes_path: str, edges_path: str) -> None:
        """Load graph data from two CSV files produced by :meth:`export_csv`.

        Args:
            nodes_path: Path to the nodes CSV.
            edges_path: Path to the edges CSV.
        """
        with open(nodes_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                nid = row["id"]
                if not self._g.node_exists(nid):
                    props = json.loads(row.get("properties") or "{}")
                    self._g.add_node(nid, row.get("label", ""), **props)

        with open(edges_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                props = json.loads(row.get("properties") or "{}")
                self._g.add_edge(
                    row["src"],
                    row["dst"],
                    row.get("label", ""),
                    weight=float(row.get("weight", 1.0)),
                    **props,
                )

    # ------------------------------------------------------------------
    # NetworkX
    # ------------------------------------------------------------------

    def to_networkx(self) -> Any:
        """Return the graph as a ``networkx.DiGraph``.

        Requires ``networkx`` (installed with ``graphvec[viz]``).

        Returns:
            A ``networkx.DiGraph`` with node/edge attributes mirroring
            the graphvec property graph.

        Raises:
            ImportError: If networkx is not installed.
        """
        try:
            import networkx as nx  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "networkx is required for to_networkx(). "
                "Install with: pip install graphvec[viz]"
            ) from exc

        G = nx.DiGraph()
        for node in self._g.nodes():
            G.add_node(node.id, label=node.label, **node.properties)
        for edge in self._g.edges():
            G.add_edge(
                edge.src,
                edge.dst,
                key=edge.id,
                label=edge.label,
                weight=edge.weight,
                **edge.properties,
            )
        return G

    def from_networkx(self, nx_graph: Any) -> None:
        """Ingest a NetworkX graph into graphvec.

        Args:
            nx_graph: A ``networkx.DiGraph`` or ``networkx.Graph``.
        """
        for nid, attrs in nx_graph.nodes(data=True):
            nid_str = str(nid)
            if not self._g.node_exists(nid_str):
                label = str(attrs.get("label", "Node"))
                props = {k: v for k, v in attrs.items() if k != "label"}
                self._g.add_node(nid_str, label, **props)

        for src, dst, attrs in nx_graph.edges(data=True):
            label = str(attrs.get("label", "CONNECTED_TO"))
            weight = float(attrs.get("weight", 1.0))
            props = {k: v for k, v in attrs.items() if k not in ("label", "weight")}
            self._g.add_edge(str(src), str(dst), label, weight=weight, **props)
