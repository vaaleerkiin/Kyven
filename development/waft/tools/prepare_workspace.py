"""Restore the development package from the deployed, self-contained plugin."""
from pathlib import Path
import shutil


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root.parents[1] / 'plugins/Cattery/WAFT'
    target = root / 'WAFT/Cattery/WAFT'
    for filename in ('WAFT.cat', 'WAFT_FP32.cat', 'WAFT.gizmo', 'cattery_waft.py'):
        if not (source / filename).is_file():
            raise FileNotFoundError(source / filename)
    # Preserve edited development files; restore only missing package files.
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.is_file() and not (target / item.name).exists():
            shutil.copy2(item, target / item.name)
    (root / 'build').mkdir(exist_ok=True)
    print('Development package ready:', target)


if __name__ == '__main__':
    main()
