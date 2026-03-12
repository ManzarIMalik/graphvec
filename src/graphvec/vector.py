"""Vector embedding storage and similarity search.

Pure-Python / stdlib fallback is always available.  When ``numpy`` is
installed (``graphvec[vector]``) computations are significantly faster.
When ``faiss-cpu`` is installed (``graphvec[faiss]``) approximate
nearest-neighbour search is available for large graphs.
"""

from __future__ import annotations

import math
import struct
import time
from typing import TYPE_CHECKING, Any

from graphvec.exceptions import EmbeddingNotFound, StorageError
from graphvec.models import Node, SearchResult

if TYPE_CHECKING:
    from graphvec.storage.base import StorageBackend

# ---------------------------------------------------------------------------
# Vector serialisation helpers (stdlib-only, no numpy required)
# ---------------------------------------------------------------------------

def _encode(vector: list[float]) -> bytes:
    """Serialise a float list to a compact binary BLOB."""
    return struct.pack(f"{len(vector)}f", *vector)


def _decode(blob: bytes) -> list[float]:
    """Deserialise a BLOB produced by :func:`_encode`."""
    n = len(blob) // 4  # 4 bytes per float32
    return list(struct.unpack(f"{n}f", blob))


# ---------------------------------------------------------------------------
# Pure-Python similarity functions
# ---------------------------------------------------------------------------

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    na, nb = _norm(a), _norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return _dot(a, b) / (na * nb)


def _euclidean_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=False)))


def _dot_product(a: list[float], b: list[float]) -> float:
    return _dot(a, b)


# ---------------------------------------------------------------------------
# Optional numpy acceleration
# ---------------------------------------------------------------------------

try:
    import numpy as np  # type: ignore[import-untyped]

    _HAS_NUMPY = True

    def _encode(vector: list[float]) -> bytes:  # type: ignore[no-redef]
        return np.array(vector, dtype=np.float32).tobytes()

    def _decode(blob: bytes) -> list[float]:  # type: ignore[no-redef]
        return np.frombuffer(blob, dtype=np.float32).tolist()

    def _cosine_similarity(a: list[float], b: list[float]) -> float:  # type: ignore[no-redef]
        av, bv = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
        na, nb = np.linalg.norm(av), np.linalg.norm(bv)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(av, bv) / (na * nb))

    def _euclidean_distance(a: list[float], b: list[float]) -> float:  # type: ignore[no-redef]
        av, bv = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
        return float(np.linalg.norm(av - bv))

    def _dot_product(a: list[float], b: list[float]) -> float:  # type: ignore[no-redef]
        av, bv = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
        return float(np.dot(av, bv))

except ImportError:
    _HAS_NUMPY = False


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class VectorStore:
    """Manages embedding storage and similarity search for one collection.

    Args:
        backend: The storage backend used to persist embeddings.
        collection: The collection namespace to operate within.
    """

    def __init__(self, backend: StorageBackend, collection: str) -> None:
        self._backend = backend
        self._collection = collection

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def set_embedding(
        self,
        node_id: str,
        vector: list[float],
        model: str = "custom",
    ) -> None:
        """Store (or replace) an embedding for *node_id*.

        Args:
            node_id: The node to attach the embedding to.
            vector: The raw float embedding vector.
            model: A human-readable label for the embedding model used.
        """
        blob = _encode(vector)
        row: dict[str, Any] = {
            "node_id": node_id,
            "vector": blob,
            "model": model,
            "dimensions": len(vector),
            "created_at": time.time(),
        }
        self._backend.insert_embedding(self._collection, row)

    def get_embedding(self, node_id: str) -> list[float]:
        """Return the embedding vector for *node_id*.

        Raises:
            EmbeddingNotFound: If no embedding is stored for the node.
        """
        row = self._backend.fetch_embedding(self._collection, node_id)
        if row is None:
            raise EmbeddingNotFound(node_id)
        return _decode(row["vector"])

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        k: int = 5,
        metric: str = "cosine",
        label_filter: str | None = None,
        node_rows: list[dict[str, Any]] | None = None,
    ) -> list[SearchResult]:
        """Return the top-*k* most similar nodes to *query_vector*.

        Args:
            query_vector: The query embedding.
            k: Number of results to return.
            metric: Distance metric — ``"cosine"`` (default),
                    ``"euclidean"``, or ``"dot"``.
            label_filter: If given, only nodes with this label are
                          considered.
            node_rows: Pre-fetched node rows (avoids a second DB query).
                       Injected by :class:`~graphvec.graph.Graph`.

        Returns:
            A list of :class:`~graphvec.models.SearchResult` ordered by
            descending similarity (or ascending distance for euclidean).
        """
        all_embs = self._backend.fetch_all_embeddings(self._collection)
        if not all_embs:
            return []

        # Build node-id → node-row lookup
        if node_rows is not None:
            node_map: dict[str, dict[str, Any]] = {r["id"]: r for r in node_rows}
        else:
            node_map = {}

        scored: list[tuple[float, dict[str, Any]]] = []
        for emb in all_embs:
            nid = emb["node_id"]
            # Apply label filter
            if label_filter is not None:
                nr = node_map.get(nid)
                if nr is None:
                    nr = self._backend.fetch_node(self._collection, nid)
                    if nr:
                        node_map[nid] = nr
                if nr is None or nr.get("label") != label_filter:
                    continue

            candidate = _decode(emb["vector"])
            if metric == "cosine":
                score = _cosine_similarity(query_vector, candidate)
                scored.append((score, emb))
            elif metric == "euclidean":
                # Negate so we can sort descending (lower dist = better)
                score = -_euclidean_distance(query_vector, candidate)
                scored.append((score, emb))
            elif metric == "dot":
                score = _dot_product(query_vector, candidate)
                scored.append((score, emb))
            else:
                raise StorageError(
                    f"Unknown metric {metric!r}. "
                    "Choose from 'cosine', 'euclidean', 'dot'."
                )

        scored.sort(key=lambda t: t[0], reverse=True)
        top = scored[:k]

        results: list[SearchResult] = []
        for raw_score, emb in top:
            nid = emb["node_id"]
            nr = node_map.get(nid) or self._backend.fetch_node(self._collection, nid)
            if nr is None:
                continue
            display_score = raw_score if metric != "euclidean" else -raw_score
            results.append(SearchResult(node=Node.from_row(nr), score=display_score, metric=metric))

        return results
