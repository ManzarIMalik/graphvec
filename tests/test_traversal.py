"""Tests for the fluent traversal API."""

from __future__ import annotations

import pytest

from graphvec import GraphVec


@pytest.fixture
def social():
    """Small social graph: alice -KNOWS-> bob -KNOWS-> carol -KNOWS-> dave."""
    db = GraphVec(":memory:")
    db.add_nodes([
        {"id": "alice", "label": "Person", "name": "Alice", "active": True},
        {"id": "bob",   "label": "Person", "name": "Bob",   "active": True},
        {"id": "carol", "label": "Person", "name": "Carol", "active": False},
        {"id": "dave",  "label": "Person", "name": "Dave",  "active": True},
        {"id": "x",     "label": "Bot",    "name": "X",     "active": True},
    ])
    db.add_edges([
        {"src": "alice", "dst": "bob",   "label": "KNOWS"},
        {"src": "bob",   "dst": "carol", "label": "KNOWS"},
        {"src": "carol", "dst": "dave",  "label": "KNOWS"},
        {"src": "alice", "dst": "x",     "label": "FOLLOWS"},
    ])
    return db


# ------------------------------------------------------------------
# Seed
# ------------------------------------------------------------------

def test_v_all(social):
    assert social.v().count() == 5


def test_v_by_id(social):
    t = social.v("alice")
    assert t.count() == 1
    assert t.first().id == "alice"


def test_v_by_label(social):
    t = social.v(label="Person")
    assert t.count() == 4


def test_v_missing_id(social):
    assert social.v("ghost").count() == 0


# ------------------------------------------------------------------
# Traversal steps
# ------------------------------------------------------------------

def test_out_one_hop(social):
    ids = social.v("alice").out("KNOWS").ids()
    assert ids == ["bob"]


def test_out_two_hops(social):
    ids = social.v("alice").out("KNOWS").out("KNOWS").ids()
    assert "carol" in ids


def test_out_hops_param(social):
    ids = social.v("alice").out("KNOWS", hops=2).ids()
    assert "carol" in ids


def test_out_no_label(social):
    ids = social.v("alice").out().ids()
    assert set(ids) == {"bob", "x"}


def test_in_(social):
    ids = social.v("bob").in_("KNOWS").ids()
    assert ids == ["alice"]


def test_both(social):
    ids = social.v("bob").both("KNOWS").ids()
    assert set(ids) == {"alice", "carol"}


# ------------------------------------------------------------------
# Filters
# ------------------------------------------------------------------

def test_has_filter(social):
    nodes = social.v(label="Person").has(active=True).all()
    names = {n["name"] for n in nodes}
    assert "Carol" not in names


def test_has_label(social):
    nodes = social.v().has_label("Bot").all()
    assert len(nodes) == 1
    assert nodes[0].id == "x"


def test_has_not(social):
    nodes = social.v(label="Person").has_not(active=False).all()
    names = {n["name"] for n in nodes}
    assert "Carol" not in names
    assert "Alice" in names


def test_where(social):
    nodes = social.v(label="Person").where(lambda n: n["name"].startswith("A")).all()
    assert len(nodes) == 1
    assert nodes[0].id == "alice"


# ------------------------------------------------------------------
# Pagination
# ------------------------------------------------------------------

def test_limit(social):
    assert social.v().limit(2).count() == 2


def test_skip(social):
    all_ids = social.v().ids()
    skipped = social.v().skip(1).ids()
    assert skipped == all_ids[1:]


def test_skip_and_limit(social):
    result = social.v().skip(1).limit(2).ids()
    assert len(result) == 2


# ------------------------------------------------------------------
# Terminal operations
# ------------------------------------------------------------------

def test_first_exists(social):
    node = social.v("alice").first()
    assert node is not None
    assert node.id == "alice"


def test_first_empty(social):
    assert social.v("ghost").first() is None


def test_ids(social):
    ids = social.v("alice").out("KNOWS").ids()
    assert ids == ["bob"]


# ------------------------------------------------------------------
# Path finding
# ------------------------------------------------------------------

def test_shortest_path(social):
    p = social.path("alice", "carol")
    assert p is not None
    assert p.nodes[0].id == "alice"
    assert p.nodes[-1].id == "carol"
    assert p.length == 2


def test_path_no_route(social):
    p = social.path("dave", "alice")
    assert p is None


def test_all_paths(social):
    paths = social.all_paths("alice", "carol", max_hops=4)
    assert len(paths) >= 1


def test_neighbors(social):
    nbrs = {n.id for n in social.neighbors("alice", hops=1)}
    assert "bob" in nbrs
    assert "x" in nbrs

    nbrs2 = {n.id for n in social.neighbors("alice", hops=2)}
    assert "carol" in nbrs2


# ------------------------------------------------------------------
# to_dataframe (optional dep guarded)
# ------------------------------------------------------------------

def test_to_dataframe_no_pandas(social, monkeypatch):
    import sys
    # Temporarily hide pandas
    saved = sys.modules.pop("pandas", None)
    try:
        with pytest.raises(ImportError, match="pandas"):
            social.v().to_dataframe()
    finally:
        if saved is not None:
            sys.modules["pandas"] = saved
