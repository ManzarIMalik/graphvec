"""Tests for JSON, CSV import/export and index management."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from graphvec import GraphVec


@pytest.fixture
def populated():
    db = GraphVec(":memory:")
    db.add_node("n1", label="Person", name="Alice", age=30)
    db.add_node("n2", label="Person", name="Bob",   age=25)
    db.add_node("n3", label="Place",  city="NYC")
    db.add_edge("n1", "n2", label="KNOWS", since=2020)
    db.add_edge("n1", "n3", label="LIVES_IN")
    return db


# ------------------------------------------------------------------
# JSON round-trip
# ------------------------------------------------------------------

def test_export_import_json_roundtrip(populated):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        populated.export_json(path)

        db2 = GraphVec(":memory:")
        db2.import_json(path)

        assert db2.node_count() == populated.node_count()
        assert db2.edge_count() == populated.edge_count()

        alice = db2.get_node("n1")
        assert alice is not None
        assert alice["name"] == "Alice"
    finally:
        os.unlink(path)


def test_export_json_structure(populated):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name
    try:
        populated.export_json(path)
        with open(path) as fh:
            data = json.load(fh)
        assert "nodes" in data
        assert "edges" in data
        assert data["version"] == 1
        assert len(data["nodes"]) == 3
        assert len(data["edges"]) == 2
    finally:
        os.unlink(path)


def test_import_json_skips_existing(populated):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        populated.export_json(path)
        # Import into the same db — should not duplicate
        populated.import_json(path)
        assert populated.node_count() == 3
    finally:
        os.unlink(path)


# ------------------------------------------------------------------
# CSV round-trip
# ------------------------------------------------------------------

def test_export_import_csv_roundtrip(populated):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as fn:
        nodes_path = fn.name
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as fe:
        edges_path = fe.name
    try:
        populated.export_csv(nodes_path, edges_path)

        db2 = GraphVec(":memory:")
        db2.import_csv(nodes_path, edges_path)

        assert db2.node_count() == populated.node_count()
        assert db2.edge_count() == populated.edge_count()
        alice = db2.get_node("n1")
        assert alice["name"] == "Alice"
    finally:
        os.unlink(nodes_path)
        os.unlink(edges_path)


# ------------------------------------------------------------------
# Subgraph export
# ------------------------------------------------------------------

def test_subgraph_export_json(populated):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        populated.subgraph(["n1", "n2"]).export_json(path)
        with open(path) as fh:
            data = json.load(fh)
        node_ids = {n["id"] for n in data["nodes"]}
        assert node_ids == {"n1", "n2"}
        # Edge n1->n2 should be included; n1->n3 should not
        assert len(data["edges"]) == 1
        assert data["edges"][0]["src"] == "n1"
        assert data["edges"][0]["dst"] == "n2"
    finally:
        os.unlink(path)


# ------------------------------------------------------------------
# Index management
# ------------------------------------------------------------------

def test_create_list_drop_index(populated):
    populated.create_index("nodes", "label")
    indexes = populated.list_indexes()
    assert any(i["field"] == "label" for i in indexes)

    populated.drop_index("nodes", "label")
    indexes = populated.list_indexes()
    assert not any(i["field"] == "label" for i in indexes)


def test_create_json_property_index(populated):
    populated.create_index("nodes", "properties.age")
    indexes = populated.list_indexes()
    assert any("age" in i["field"] for i in indexes)


def test_create_edge_index(populated):
    populated.create_index("edges", "label")
    indexes = populated.list_indexes()
    assert any(i["target"] == "edges" for i in indexes)


def test_invalid_index_target(populated):
    from graphvec.exceptions import StorageError
    with pytest.raises(StorageError):
        populated.create_index("unknown_table", "field")


# ------------------------------------------------------------------
# NetworkX interop (guarded — only if networkx available)
# ------------------------------------------------------------------

def test_to_networkx_no_networkx(populated, monkeypatch):
    import sys
    saved = sys.modules.pop("networkx", None)
    try:
        with pytest.raises(ImportError, match="networkx"):
            populated.to_networkx()
    finally:
        if saved is not None:
            sys.modules["networkx"] = saved


def test_from_networkx_no_networkx(populated, monkeypatch):
    import sys
    saved = sys.modules.pop("networkx", None)
    try:
        with pytest.raises((ImportError, AttributeError)):
            populated.from_networkx(object())
    finally:
        if saved is not None:
            sys.modules["networkx"] = saved
