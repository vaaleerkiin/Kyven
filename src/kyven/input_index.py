"""Lightweight shared input inventory; one SQLite database per cache work root."""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TypeVar

from kyven.cache_layout import work_root

_active: ContextVar[dict | None] = ContextVar("input_index_connections", default=None)
_lock = threading.RLock()
_Result = TypeVar("_Result")


@contextmanager
def batch() -> Iterator[None]:
    connections: dict[Path, sqlite3.Connection] = {}
    token = _active.set(connections)
    try:
        yield
    finally:
        _active.reset(token)
        for connection in connections.values():
            connection.close()


@contextmanager
def _connection(path: Path, create: bool = False) -> Iterator[sqlite3.Connection | None]:
    database = work_root(path) / "inputs.sqlite3"
    if not create and not database.is_file():
        yield None
        return
    connections = _active.get()
    existing = connections.get(database) if connections is not None else None
    if existing is not None:
        yield existing
        return
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database, timeout=10)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("CREATE TABLE IF NOT EXISTS inputs (path TEXT PRIMARY KEY, record TEXT NOT NULL)")
        if connections is not None:
            connections[database] = connection
        yield connection
    finally:
        if connections is None or connections.get(database) is not connection:
            connection.close()


def _recover(path: Path) -> None:
    """Quarantine a corrupt disposable index, retaining media and diagnostics."""
    database = work_root(path) / "inputs.sqlite3"
    connections = _active.get()
    if connections is not None:
        connection = connections.pop(database, None)
        if connection is not None:
            connection.close()
    backup = database.with_name(f"{database.name}.corrupt-{uuid.uuid4().hex}")
    # Never treat busy/locked databases as corruption. On Windows an index held by
    # another process cannot be moved; surface that failure rather than delete it.
    for suffix in ("", "-wal", "-shm"):
        source = database.with_name(database.name + suffix)
        if source.exists():
            source.replace(backup.with_name(backup.name + suffix))


def _run(path: Path, operation: Callable[[], _Result]) -> _Result:
    with _lock:
        try:
            return operation()
        except sqlite3.DatabaseError as exc:
            code = getattr(exc, "sqlite_errorcode", 0) & 0xff
            corrupt = code in {11, 26}  # SQLITE_CORRUPT / SQLITE_NOTADB, including extended codes.
            # sqlite_errorcode is exposed by Python 3.11+, while Kyven also supports 3.10.
            if not corrupt and str(exc) not in {
                "file is not a database", "database disk image is malformed",
                "file is encrypted or is not a database",
            }:
                raise
            _recover(path)
            return operation()


def read(path: Path) -> dict | None:
    def operation():
        with _connection(path) as connection:
            if connection is None:
                return None
            row = connection.execute("SELECT record FROM inputs WHERE path=?", (str(path.resolve()),)).fetchone()
            return json.loads(row[0]) if row else None
    return _run(path, operation)


def write(path: Path, record: dict) -> None:
    def operation():
        with _connection(path, create=True) as connection, connection:
            connection.execute("INSERT OR REPLACE INTO inputs VALUES (?, ?)",
                               (str(path.resolve()), json.dumps(record, sort_keys=True)))
    _run(path, operation)


def forget(path: Path) -> None:
    def operation():
        with _connection(path) as connection:
            if connection is not None:
                with connection:
                    connection.execute("DELETE FROM inputs WHERE path=?", (str(path.resolve()),))
    _run(path, operation)
