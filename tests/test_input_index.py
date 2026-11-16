"""Disposable index recovery must preserve media and distinguish corruption from locks."""
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from kyven import input_index
from kyven.cache_layout import work_root


@pytest.mark.parametrize('action', ['read', 'write', 'forget'])
@pytest.mark.parametrize('batched', [False, True])
def test_corrupt_database_is_quarantined_and_operations_resume(tmp_path, action, batched):
    path = tmp_path/'_kyven_work/source.1001.tif'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'keep source pixels')
    database = work_root(path)/'inputs.sqlite3'
    database.write_bytes(b'broken database')
    before = path.read_bytes()

    def operate():
        if action == 'read':
            assert input_index.read(path) is None
        elif action == 'write':
            input_index.write(path, {'frame': 1001})
        else:
            input_index.forget(path)
        input_index.write(path, {'frame': 1002})
        assert input_index.read(path) == {'frame': 1002}
    if batched:
        with input_index.batch():
            operate()
    else:
        operate()
    assert path.read_bytes() == before
    backups = list(path.parent.glob('inputs.sqlite3.corrupt-*'))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b'broken database'


def test_nested_batches_restore_outer_connection_and_isolate_roots(tmp_path):
    a, b = tmp_path/'a/source.tif', tmp_path/'b/source.tif'
    with input_index.batch():
        input_index.write(a, {'frame': 1})
        with input_index.batch():
            input_index.write(b, {'frame': 2})
        assert input_index.read(a) == {'frame': 1}
        assert input_index.read(b) == {'frame': 2}
        input_index.forget(a)
        assert input_index.read(a) is None
    assert input_index.read(b) == {'frame': 2}


def test_parallel_writes_keep_independent_frame_receipts(tmp_path):
    def store(frame):
        path = tmp_path/f'source.{frame}.tif'
        input_index.write(path, {'frame': frame})
        assert input_index.read(path) == {'frame': frame}
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(store, range(20)))
    assert not list(tmp_path.rglob('*.corrupt-*'))


@pytest.mark.parametrize('message', ['database is locked', 'attempt to write a readonly database',
                                    'disk I/O error'])
def test_lock_and_io_errors_do_not_quarantine_valid_database(tmp_path, monkeypatch, message):
    path = tmp_path/'source.tif'
    input_index.write(path, {'frame': 1})
    database = work_root(path)/'inputs.sqlite3'
    before = database.read_bytes()
    monkeypatch.setattr(input_index.sqlite3, 'connect',
        lambda *args, **kwargs: (_ for _ in ()).throw(sqlite3.OperationalError(message)))
    with pytest.raises(sqlite3.OperationalError, match=message):
        input_index.read(path)
    assert database.read_bytes() == before
    assert not list(database.parent.glob('*.corrupt-*'))


def test_malformed_receipt_is_not_database_corruption(tmp_path):
    path = tmp_path/'source.tif'
    input_index.write(path, {'frame': 1})
    database = work_root(path)/'inputs.sqlite3'
    with sqlite3.connect(database) as connection:
        connection.execute('UPDATE inputs SET record=?', ('invalid json',))
    with pytest.raises(json.JSONDecodeError):
        input_index.read(path)
    input_index.forget(path)
    input_index.write(path, {'frame': 2})
    assert input_index.read(path) == {'frame': 2}
    assert not list(database.parent.glob('*.corrupt-*'))
