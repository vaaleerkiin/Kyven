"""Lightweight shared input inventory; one SQLite database per cache work root."""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from kyven.cache_layout import work_root

_active: ContextVar[dict | None] = ContextVar("input_index_connections", default=None)


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
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("CREATE TABLE IF NOT EXISTS inputs (path TEXT PRIMARY KEY, record TEXT NOT NULL)")
    if connections is not None:
        connections[database] = connection
    try:
        yield connection
    finally:
        if connections is None:
            connection.close()


def read(path: Path) -> dict | None:
    with _connection(path) as connection:
        if connection is None:
            return None
        row = connection.execute("SELECT record FROM inputs WHERE path=?", (str(path.resolve()),)).fetchone()
        return json.loads(row[0]) if row else None


def write(path: Path, record: dict) -> None:
    with _connection(path, create=True) as connection, connection:
        connection.execute("INSERT OR REPLACE INTO inputs VALUES (?, ?)",
                           (str(path.resolve()), json.dumps(record, sort_keys=True)))


def forget(path: Path) -> None:
    with _connection(path) as connection:
        if connection is not None:
            with connection:
                connection.execute("DELETE FROM inputs WHERE path=?", (str(path.resolve()),))
