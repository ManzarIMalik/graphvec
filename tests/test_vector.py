"""Tests for vector embedding storage and similarity search."""

from __future__ import annotations

import math

import pytest

from graphvec import EmbeddingNotFound, GraphVec


@pytest.fixture
def db():
    g = GraphVec(":memory:")
    g.add_node("n1", label="Doc", text="hello world")
    g.add_node("n2", label="Doc", text="foo bar")
    g.add_node("n3", label="Other", text="baz")
    return g


def _vec(values):
    """Normalise a vector to unit length."""
    mag = math.sqrt(sum(v * v for v in values))
    return [v / mag for v in values]


# ------------------------------------------------------------------
# Embedding storage
# ------------------------------------------------------------------

def test_set_and_get_embedding(db):
    vec = _vec([1.0, 0.0, 0.0])
    db.set_embedding("n1", vec)
    retrieved = db.get_embedding("n1")
    assert len(retrieved) == 3
    assert abs(retrieved[0] - 1.0) < 1e-5


def test_get_embedding_missing(db):
    with pytest.raises(EmbeddingNotFound):
        db.get_embedding("n1")  # not set yet


def test_set_embedding_missing_node(db):
    from graphvec import NodeNotFound
    with pytest.raises(NodeNotFound):
        db.set_embedding("ghost", [1.0, 0.0])


def test_replace_embedding(db):
    vec1 = _vec([1.0, 0.0, 0.0])
    vec2 = _vec([0.0, 1.0, 0.0])
    db.set_embedding("n1", vec1)
    db.set_embedding("n1", vec2)
    retrieved = db.get_embedding("n1")
    assert abs(retrieved[1] - 1.0) < 1e-5


# ------------------------------------------------------------------
# Cosine search
# ------------------------------------------------------------------

def test_search_cosine_returns_top_k(db):
    db.set_embedding("n1", _vec([1.0, 0.0]))
    db.set_embedding("n2", _vec([0.0, 1.0]))
    db.set_embedding("n3", _vec([1.0, 1.0]))

    results = list(db.search(_vec([1.0, 0.0]), k=2))
    assert len(results) == 2
    assert results[0].node.id == "n1"
    assert results[0].metric == "cosine"


def test_search_returns_score(db):
    db.set_embedding("n1", _vec([1.0, 0.0]))
    results = list(db.search(_vec([1.0, 0.0]), k=1))
    assert abs(results[0].score - 1.0) < 1e-5


def test_search_label_filter(db):
    db.set_embedding("n1", _vec([1.0, 0.0]))
    db.set_embedding("n2", _vec([0.9, 0.1]))
    db.set_embedding("n3", _vec([0.8, 0.2]))

    results = list(db.search(_vec([1.0, 0.0]), k=10, label="Doc"))
    ids = [r.node.id for r in results]
    assert "n3" not in ids  # n3 is label "Other"
    assert "n1" in ids


# ------------------------------------------------------------------
# Euclidean search
# ------------------------------------------------------------------

def test_search_euclidean(db):
    db.set_embedding("n1", [1.0, 0.0])
    db.set_embedding("n2", [0.0, 1.0])
    db.set_embedding("n3", [0.9, 0.1])

    results = list(db.search([1.0, 0.0], k=1, metric="euclidean"))
    assert results[0].node.id == "n1"
    assert results[0].score >= 0  # score = distance (lower = better)


# ------------------------------------------------------------------
# Dot product search
# ------------------------------------------------------------------

def test_search_dot(db):
    db.set_embedding("n1", [2.0, 0.0])
    db.set_embedding("n2", [0.0, 1.0])

    results = list(db.search([1.0, 0.0], k=1, metric="dot"))
    assert results[0].node.id == "n1"


# ------------------------------------------------------------------
# Unknown metric
# ------------------------------------------------------------------

def test_search_unknown_metric(db):
    db.set_embedding("n1", [1.0, 0.0])
    from graphvec import StorageError
    with pytest.raises(StorageError, match="Unknown metric"):
        list(db.search([1.0, 0.0], k=1, metric="manhattan"))


# ------------------------------------------------------------------
# Empty search
# ------------------------------------------------------------------

def test_search_no_embeddings(db):
    results = list(db.search([1.0, 0.0], k=5))
    assert results == []


# ------------------------------------------------------------------
# Auto-embedding
# ------------------------------------------------------------------

def test_auto_embedding_on_insert():
    calls = []

    def embed_fn(text: str) -> list[float]:
        calls.append(text)
        return _vec([1.0, 0.0, 0.0])

    db = GraphVec(":memory:", embed_fn=embed_fn, embed_field="content")
    db.add_node("n1", label="Doc", content="hello world")
    assert len(calls) == 1
    assert calls[0] == "hello world"
    vec = db.get_embedding("n1")
    assert len(vec) == 3


def test_auto_embedding_skipped_when_field_absent():
    calls = []

    def embed_fn(text: str) -> list[float]:
        calls.append(text)
        return [1.0]

    db = GraphVec(":memory:", embed_fn=embed_fn, embed_field="content")
    db.add_node("n1", label="Doc", title="no content field")
    assert len(calls) == 0


# ------------------------------------------------------------------
# search_text requires embed_fn
# ------------------------------------------------------------------

def test_search_text_no_embed_fn():
    db = GraphVec(":memory:")
    from graphvec import StorageError
    with pytest.raises(StorageError, match="embed_fn"):
        db.search_text("hello")


def test_search_text_with_embed_fn():
    def embed_fn(text: str) -> list[float]:
        return _vec([1.0, 0.0])

    db = GraphVec(":memory:", embed_fn=embed_fn)
    db.add_node("n1", label="Doc", text="hello")
    db.set_embedding("n1", _vec([1.0, 0.0]))

    results = list(db.search_text("hello", k=1))
    assert len(results) == 1


# ------------------------------------------------------------------
# Hybrid: search + traversal
# ------------------------------------------------------------------

def test_search_then_traverse():
    db = GraphVec(":memory:")
    db.add_node("n1", label="Claim")
    db.add_node("n2", label="Evidence")
    db.add_node("n3", label="Claim")
    db.add_edge("n1", "n2", label="SUPPORTED_BY")
    db.set_embedding("n1", _vec([1.0, 0.0]))
    db.set_embedding("n3", _vec([0.0, 1.0]))

    results = db.search(_vec([1.0, 0.0]), k=1).out("SUPPORTED_BY").all()
    assert len(results) == 1
    assert results[0].id == "n2"
