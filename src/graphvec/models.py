"""Typed dataclasses for all public return objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Node:
    """A node in the property graph.

    Attributes:
        id: Unique identifier for the node.
        label: Semantic label / type (e.g. ``"Person"``, ``"Document"``).
        properties: Arbitrary key-value metadata stored with the node.
        created_at: Unix timestamp of creation.
        updated_at: Unix timestamp of last update.
    """

    id: str
    label: str
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style property access: ``node["name"]``."""
        return self.properties[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Return property value or *default* if missing."""
        return self.properties.get(key, default)

    def __contains__(self, key: str) -> bool:
        """Support ``"key" in node`` checks."""
        return key in self.properties

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Node:
        """Construct a Node from a raw storage row dict."""
        return cls(
            id=row["id"],
            label=row.get("label", ""),
            properties=row.get("properties", {}),
            created_at=row.get("created_at", 0.0),
            updated_at=row.get("updated_at", 0.0),
        )


@dataclass
class Edge:
    """A directed, typed edge between two nodes.

    Attributes:
        id: Unique identifier for the edge.
        src: Source node id.
        dst: Destination node id.
        label: Relationship type (e.g. ``"KNOWS"``).
        properties: Arbitrary key-value metadata stored with the edge.
        weight: Numeric edge weight (default ``1.0``).
        created_at: Unix timestamp of creation.
        updated_at: Unix timestamp of last update.
    """

    id: str
    src: str
    dst: str
    label: str
    properties: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    created_at: float = 0.0
    updated_at: float = 0.0

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style property access."""
        return self.properties[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Return property value or *default* if missing."""
        return self.properties.get(key, default)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Edge:
        """Construct an Edge from a raw storage row dict."""
        return cls(
            id=row["id"],
            src=row["src"],
            dst=row["dst"],
            label=row.get("label", ""),
            properties=row.get("properties", {}),
            weight=row.get("weight", 1.0),
            created_at=row.get("created_at", 0.0),
            updated_at=row.get("updated_at", 0.0),
        )


@dataclass
class SearchResult:
    """A single result from a vector similarity search.

    Attributes:
        node: The matching :class:`Node`.
        score: Similarity score (higher is more similar for cosine/dot;
               lower is more similar for euclidean).
        metric: The distance metric used (``"cosine"``, ``"euclidean"``,
                ``"dot"``).
    """

    node: Node
    score: float
    metric: str


@dataclass
class Path:
    """A path through the graph between two nodes.

    Attributes:
        nodes: Ordered list of nodes along the path (start → end).
        edges: Ordered list of edges traversed.
        length: Number of hops (``len(edges)``).
    """

    nodes: list[Node]
    edges: list[Edge]
    length: int

    def __post_init__(self) -> None:
        self.length = len(self.edges)
