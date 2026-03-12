"""Tests for transaction context manager and manual transactions."""

from __future__ import annotations

import pytest

from graphvec import GraphVec


@pytest.fixture
def db():
    return GraphVec(":memory:")


# ------------------------------------------------------------------
# Context-manager form
# ------------------------------------------------------------------

def test_transaction_commit(db):
    with db.transaction():
        db.add_node("n1", label="X")
        db.add_node("n2", label="Y")
        db.add_edge("n1", "n2", label="Z")

    assert db.node_count() == 2
    assert db.edge_count() == 1


def test_transaction_rollback_on_exception(db):
    """Nodes inserted inside a failed transaction must not be persisted."""
    try:
        with db.transaction():
            db.add_node("n1", label="X")
            raise RuntimeError("oops")
    except RuntimeError:
        pass

    assert not db.node_exists("n1"), "n1 should have been rolled back"


def test_nested_operations_atomic(db):
    """All nodes added within a transaction block should persist together."""
    with db.transaction():
        db.add_node("a", label="A")
        db.add_node("b", label="B")

    assert db.node_exists("a")
    assert db.node_exists("b")


# ------------------------------------------------------------------
# Manual transaction API
# ------------------------------------------------------------------

def test_manual_commit(db):
    txn = db.begin()
    db.add_node("n1", label="X")
    txn.commit()
    assert db.node_exists("n1")


def test_manual_rollback(db):
    # Verify that Transaction.rollback() doesn't raise
    txn = db.begin()
    txn.rollback()


def test_transaction_context_manager_protocol(db):
    """Transaction object works as context manager."""
    from graphvec.transaction import Transaction
    txn = Transaction(db._backend)
    with txn:
        db.add_node("ctx1", label="X")
    assert db.node_exists("ctx1")


def test_transaction_double_commit_raises(db):
    from graphvec.exceptions import StorageError
    txn = db.begin()
    txn.commit()
    with pytest.raises(StorageError):
        txn.commit()


# ------------------------------------------------------------------
# Multiple transactions sequentially
# ------------------------------------------------------------------

def test_sequential_transactions(db):
    with db.transaction():
        db.add_node("x1", label="A")

    with db.transaction():
        db.add_node("x2", label="B")

    assert db.node_count() == 2


# ------------------------------------------------------------------
# Collection management
# ------------------------------------------------------------------

def test_collection_isolation(db):
    col1 = db.collection("col1")
    col2 = db.collection("col2")

    col1.add_node("n1", label="X")
    col2.add_node("n1", label="Y")  # same id, different collection

    assert col1.get_node("n1").label == "X"
    assert col2.get_node("n1").label == "Y"


def test_list_and_drop_collection(db):
    col = db.collection("test_col")
    col.add_node("n1", label="X")

    collections = db.list_collections()
    assert "test_col" in collections

    db.drop_collection("test_col")
    assert "test_col" not in db.list_collections()


def test_drop_nonexistent_collection(db):
    from graphvec import CollectionNotFound
    with pytest.raises(CollectionNotFound):
        db.drop_collection("does_not_exist")
