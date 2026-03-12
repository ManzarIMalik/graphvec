"""Tests for node and edge CRUD operations."""

from __future__ import annotations

import pytest

from graphvec import EdgeNotFound, GraphVec, NodeNotFound


@pytest.fixture
def db():
    return GraphVec(":memory:")


# ------------------------------------------------------------------
# Node CRUD
# ------------------------------------------------------------------

def test_add_and_get_node(db):
    node = db.add_node("n1", label="Person", name="Alice")
    assert node.id == "n1"
    assert node.label == "Person"
    assert node["name"] == "Alice"

    fetched = db.get_node("n1")
    assert fetched is not None
    assert fetched.id == "n1"


def test_get_node_missing_returns_none(db):
    assert db.get_node("missing") is None


def test_node_exists(db):
    db.add_node("n1", label="X")
    assert db.node_exists("n1")
    assert not db.node_exists("nope")


def test_update_node(db):
    db.add_node("n1", label="Person", name="Alice")
    updated = db.update_node("n1", name="Alicia", age=30)
    assert updated["name"] == "Alicia"
    assert updated["age"] == 30


def test_update_node_missing(db):
    with pytest.raises(NodeNotFound):
        db.update_node("ghost", name="X")


def test_delete_node(db):
    db.add_node("n1", label="Thing")
    db.delete_node("n1")
    assert db.get_node("n1") is None


def test_delete_node_missing(db):
    with pytest.raises(NodeNotFound):
        db.delete_node("ghost")


def test_nodes_list(db):
    db.add_node("n1", label="Person", name="Alice")
    db.add_node("n2", label="Person", name="Bob")
    db.add_node("n3", label="Place", city="NYC")

    all_nodes = db.nodes()
    assert len(all_nodes) == 3

    persons = db.nodes(label="Person")
    assert len(persons) == 2

    filtered = db.nodes(label="Person", name="Alice")
    assert len(filtered) == 1
    assert filtered[0].id == "n1"


def test_node_count(db):
    assert db.node_count() == 0
    db.add_node("n1", label="X")
    assert db.node_count() == 1


def test_bulk_add_nodes(db):
    nodes = db.add_nodes([
        {"id": "a", "label": "P", "name": "Alice"},
        {"id": "b", "label": "P", "name": "Bob"},
    ])
    assert len(nodes) == 2
    assert db.node_count() == 2


def test_node_property_access(db):
    db.add_node("n1", label="X", score=0.9, tags=["a", "b"])
    node = db.get_node("n1")
    assert node["score"] == 0.9
    assert node.get("missing", "default") == "default"
    assert "score" in node


# ------------------------------------------------------------------
# Edge CRUD
# ------------------------------------------------------------------

def test_add_and_get_edge(db):
    db.add_node("n1", label="A")
    db.add_node("n2", label="B")
    edge = db.add_edge("n1", "n2", label="KNOWS", since=2020)
    assert edge.src == "n1"
    assert edge.dst == "n2"
    assert edge.label == "KNOWS"
    assert edge["since"] == 2020

    fetched = db.get_edge(edge.id)
    assert fetched is not None
    assert fetched.id == edge.id


def test_add_edge_missing_src(db):
    db.add_node("n2", label="B")
    with pytest.raises(NodeNotFound):
        db.add_edge("ghost", "n2", label="X")


def test_add_edge_missing_dst(db):
    db.add_node("n1", label="A")
    with pytest.raises(NodeNotFound):
        db.add_edge("n1", "ghost", label="X")


def test_update_edge(db):
    db.add_node("n1", label="A")
    db.add_node("n2", label="B")
    edge = db.add_edge("n1", "n2", label="KNOWS")
    updated = db.update_edge(edge.id, strength=5)
    assert updated["strength"] == 5


def test_update_edge_missing(db):
    with pytest.raises(EdgeNotFound):
        db.update_edge("ghost-id", foo="bar")


def test_delete_edge(db):
    db.add_node("n1", label="A")
    db.add_node("n2", label="B")
    edge = db.add_edge("n1", "n2", label="X")
    db.delete_edge(edge.id)
    assert db.get_edge(edge.id) is None


def test_delete_edge_missing(db):
    with pytest.raises(EdgeNotFound):
        db.delete_edge("ghost-edge")


def test_edges_filter(db):
    db.add_node("n1", label="A")
    db.add_node("n2", label="B")
    db.add_node("n3", label="C")
    db.add_edge("n1", "n2", label="KNOWS")
    db.add_edge("n2", "n3", label="LIKES")
    db.add_edge("n1", "n3", label="KNOWS")

    assert len(db.edges()) == 3
    assert len(db.edges(label="KNOWS")) == 2
    assert len(db.edges(src="n1")) == 2
    assert len(db.edges(dst="n3")) == 2


def test_edge_exists(db):
    db.add_node("n1", label="A")
    db.add_node("n2", label="B")
    db.add_edge("n1", "n2", label="X")
    assert db.edge_exists("n1", "n2")
    assert db.edge_exists("n1", "n2", label="X")
    assert not db.edge_exists("n2", "n1")


def test_edge_count(db):
    db.add_node("n1", label="A")
    db.add_node("n2", label="B")
    assert db.edge_count() == 0
    db.add_edge("n1", "n2", label="X")
    assert db.edge_count() == 1


def test_delete_node_cascades_edges(db):
    db.add_node("n1", label="A")
    db.add_node("n2", label="B")
    db.add_edge("n1", "n2", label="X")
    db.delete_node("n1")
    assert db.edge_count() == 0


def test_bulk_add_edges(db):
    db.add_node("a", label="P")
    db.add_node("b", label="P")
    db.add_node("c", label="P")
    edges = db.add_edges([
        {"src": "a", "dst": "b", "label": "KNOWS"},
        {"src": "b", "dst": "c", "label": "KNOWS"},
    ])
    assert len(edges) == 2


def test_edge_weight(db):
    db.add_node("n1", label="A")
    db.add_node("n2", label="B")
    edge = db.add_edge("n1", "n2", label="X", weight=0.5)
    assert edge.weight == 0.5
