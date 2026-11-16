"""Keep user-facing local documentation and image links usable in a checkout or ZIP."""
import re
from pathlib import Path
from urllib.parse import unquote, urlparse


def test_local_documentation_targets_exist():
    root = Path(__file__).parents[1]
    files = [root/'README.md', root/'PROJECT_BIBLE.md', *(root/'docs').rglob('*.md'),
             root/'plugins/README.md', *(root/'plugins/Cattery/WAFT').glob('*.md')]
    missing = []
    for path in files:
        for match in re.finditer(r'!?\[[^\]]*\]\(([^)\s]+)(?:\s+[^)]*)?\)',
                                 path.read_text(encoding='utf-8')):
            target = match.group(1).strip('<>')
            if urlparse(target).scheme or target.startswith('#'):
                continue
            target = unquote(target.split('#')[0])
            if target and not (path.parent/target).exists():
                missing.append(f'{path.relative_to(root)}: {target}')
    assert not missing, '\n'.join(missing)
