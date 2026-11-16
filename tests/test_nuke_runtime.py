"""Opt-in licensed Nuke contract test; ordinary tests remain host independent."""
import json
import os
import subprocess
from pathlib import Path

import pytest


def test_real_nuke_nodes_guidance_upgrades_and_reopen(tmp_path):
    executable = os.environ.get('KYVEN_NUKE_EXECUTABLE')
    if not executable:
        pytest.skip('Set KYVEN_NUKE_EXECUTABLE for licensed Nuke integration checks')
    report = tmp_path/'nuke-smoke.json'
    script = Path(__file__).parent/'nuke/smoke_nodes.py'
    result = subprocess.run([executable, '--nukex', '-i', '-t', str(script)],
        env={**os.environ, 'KYVEN_NUKE_SMOKE_REPORT': str(report)},
        capture_output=True, text=True, timeout=90, check=False,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    assert report.is_file(), result.stdout + result.stderr
    payload = json.loads(report.read_text())
    assert payload.get('ok'), json.dumps(payload, indent=2) + result.stderr
