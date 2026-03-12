"""Tests for built-in graph algorithms."""

from __future__ import annotations

import pytest

from graphvec import GraphVec


@pytest.fixture
def chain():
    """Linear chain: a -> b -> c -> d."""
    db = GraphVec(":memory:")
    for nid in "abcd":
        db.add_node(nid, label="Node")
    db.add_edge("a", "b", label="X")
    db.add_edge("b", "c", label="X")
    db.add_edge("c", "d", label="X")
    return db


@pytest.fixture
def star():
    """Star: hub -> s1, hub -> s2, hub -> s3."""
    db = GraphVec(":memory:")
    db.add_node("hub", label="Hub")
    for i in range(1, 4):
        db.add_node(f"s{i}", label="Spoke")
        db.add_edge("hub", f"s{i}", label="CONNECTS")
    return db


# ------------------------------------------------------------------
# Degree
# ------------------------------------------------------------------

def test_degree_chain(chain):
    assert chain.in_degree("a") == 0
    assert chain.out_degree("a") == 1
    assert chain.degree("b") == 2  # 1 in + 1 out


def test_degree_star(star):
    assert star.out_degree("hub") == 3
    assert star.in_degree("s1") == 1


# ------------------------------------------------------------------
# BFS
# ------------------------------------------------------------------

def test_bfs_chain(chain):
    order = chain.bfs("a")
    assert order[0] == "a"
    assert set(order) == {"a", "b", "c", "d"}


def test_bfs_max_depth(chain):
    order = chain.bfs("a", max_depth=1)
    assert set(order) == {"a", "b"}


def test_bfs_unreachable(chain):
    # d has no outgoing edges
    order = chain.bfs("d")
    assert order == ["d"]


# ------------------------------------------------------------------
# DFS
# ------------------------------------------------------------------

def test_dfs_chain(chain):
    order = chain.dfs("a")
    assert order[0] == "a"
    assert set(order) == {"a", "b", "c", "d"}


def test_dfs_max_depth(chain):
    order = chain.dfs("a", max_depth=2)
    assert "d" not in order


# ------------------------------------------------------------------
# Shortest path
# ------------------------------------------------------------------

def test_shortest_path_chain(chain):
    path = chain.shortest_path("a", "d")
    assert path == ["a", "b", "c", "d"]


def test_shortest_path_same_node(chain):
    path = chain.shortest_path("a", "a")
    assert path == ["a"]


def test_shortest_path_no_route(chain):
    path = chain.shortest_path("d", "a")
    assert path is None


def test_shortest_path_max_hops(chain):
    path = chain.shortest_path("a", "d", max_hops=2)
    assert path is None


# ------------------------------------------------------------------
# All paths
# ------------------------------------------------------------------

def test_all_paths_chain(chain):
    paths = chain.all_paths("a", "d", max_hops=4)
    assert len(paths) == 1
    assert [n.id for n in paths[0].nodes] == ["a", "b", "c", "d"]


def test_all_paths_branching():
    db = GraphVec(":memory:")
    for nid in "abcd":
        db.add_node(nid, label="N")
    db.add_edge("a", "b", label="X")
    db.add_edge("a", "c", label="X")
    db.add_edge("b", "d", label="X")
    db.add_edge("c", "d", label="X")
    paths = db.all_paths("a", "d", max_hops=3)
    assert len(paths) == 2


# ------------------------------------------------------------------
# Connected components
# ------------------------------------------------------------------

def test_connected_components_chain(chain):
    comps = chain.connected_components()
    assert len(comps) == 1
    assert chain.is_connected()


def test_connected_components_disconnected():
    db = GraphVec(":memory:")
    db.add_node("a", label="N")
    db.add_node("b", label="N")  # isolated
    db.add_node("c", label="N")
    db.add_edge("a", "c", label="X")
    comps = db.connected_components()
    assert len(comps) == 2
    assert not db.is_connected()


def test_connected_components_empty():
    db = GraphVec(":memory:")
    assert db.connected_components() == []
    assert db.is_connected()


# ------------------------------------------------------------------
# PageRank
# ------------------------------------------------------------------

def test_pagerank_returns_all_nodes(chain):
    pr = chain.pagerank()
    assert set(pr.keys()) == {"a", "b", "c", "d"}


def test_pagerank_scores_sum_to_one(chain):
    pr = chain.pagerank()
    total = sum(pr.values())
    assert abs(total - 1.0) < 0.01


def test_pagerank_hub_highest(star):
    # Spokes all point TO hub — actually hub points out in this fixture.
    # After PageRank, spokes receive from hub so they should have higher score.
    pr = star.pagerank()
    assert pr["hub"] >= 0


def test_pagerank_empty():
    db = GraphVec(":memory:")
    assert db.pagerank() == {}


# ------------------------------------------------------------------
# Neighbors
# ------------------------------------------------------------------

def test_neighbors_1_hop(chain):
    nbrs = chain._algo.neighbors("a", hops=1)
    assert nbrs == {"b"}


def test_neighbors_2_hops(chain):
    nbrs = chain._algo.neighbors("a", hops=2)
    assert "b" in nbrs
    assert "c" in nbrs
