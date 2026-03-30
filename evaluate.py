"""
Point d'entrée à la racine du dépôt : exécute le même pipeline que ``src/evaluate.py``.

Usage : ``python evaluate.py --model gpt2`` (équivalent à ``python src/evaluate.py ...``).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def main() -> None:
    path = Path(__file__).resolve().parent / "src" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("_hb_evaluate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Impossible de charger src/evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


if __name__ == "__main__":
    main()
