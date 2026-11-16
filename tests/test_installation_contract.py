"""The documented ZIP install must not introduce an unadvertised VCS requirement."""
from pathlib import Path


def test_runtime_installer_and_requirements_have_no_git_only_dependencies():
    root = Path(__file__).parents[1]
    paths = [root/'install.ps1', *list((root/'requirements').glob('*.txt'))]
    for path in paths:
        text = path.read_text(encoding='utf-8').lower()
        assert 'git+https' not in text, f'{path.name} requires Git for ZIP users'
        assert 'tapnet' not in text, f'{path.name} installs an unused development package'
