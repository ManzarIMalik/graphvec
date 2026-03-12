"""graphvec — embedded, serverless, persistent graph database with vector search.

Zero mandatory dependencies; pure-Python stdlib core.

Quick start::

    from graphvec import GraphVec

    db = GraphVec("mydb.db")
    db.add_node("alice", label="Person", name="Alice")
    db.add_node("bob",   label="Person", name="Bob")
    db.add_edge("alice", "bob", label="KNOWS")

    for node in db.v("alice").out("KNOWS").all():
        print(node.id, node["name"])
"""

from graphvec.db import GraphVec
from graphvec.exceptions import (
    CollectionNotFound,
    EdgeNotFound,
    EmbeddingNotFound,
    GraphVecError,
    NodeNotFound,
    StorageError,
)
from graphvec.models import Edge, Node, Path, SearchResult

__version__ = "0.1.0"
__all__ = [
    # Main entry point
    "GraphVec",
    # Models
    "Node",
    "Edge",
    "Path",
    "SearchResult",
    # Exceptions
    "GraphVecError",
    "NodeNotFound",
    "EdgeNotFound",
    "EmbeddingNotFound",
    "StorageError",
    "CollectionNotFound",
]
