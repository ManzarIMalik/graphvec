"""Transaction context manager and manual transaction API.

Wraps the storage backend's begin / commit / rollback primitives in a
convenient Python context manager so callers get automatic rollback on
any unhandled exception.
"""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING

from graphvec.exceptions import StorageError

if TYPE_CHECKING:
    from graphvec.storage.base import StorageBackend


class Transaction:
    """Represents an active database transaction.

    Prefer the context-manager form via :py:meth:`graphvec.graph.Graph.transaction`::

        with g.transaction():
            g.add_node("n1", label="Claim")
            g.add_edge("n1", "n2", label="SUPPORTS")

    For manual control::

        txn = g.begin()
        try:
            g.add_node("n1", label="Claim")
            txn.commit()
        except Exception:
            txn.rollback()

    Args:
        backend: The underlying storage backend.
    """

    def __init__(self, backend: StorageBackend) -> None:
        self._backend = backend
        self._open = False

    def begin(self) -> Transaction:
        """Begin the transaction and return *self* for chaining."""
        self._backend.begin()
        self._open = True
        return self

    def commit(self) -> None:
        """Commit all changes made within this transaction.

        Raises:
            StorageError: If the commit fails.
        """
        if not self._open:
            raise StorageError("Transaction is not open.")
        self._backend.commit()
        self._open = False

    def rollback(self) -> None:
        """Discard all changes made within this transaction.

        Raises:
            StorageError: If the rollback fails.
        """
        if not self._open:
            return
        self._backend.rollback()
        self._open = False

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> Transaction:
        if not self._open:
            self.begin()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        if exc_type is not None:
            self.rollback()
        else:
            try:
                self.commit()
            except StorageError:
                self.rollback()
                raise
        return False  # do not suppress exceptions
