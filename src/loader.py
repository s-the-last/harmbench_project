"""
Chargement des jeux de prompts au format JSONL depuis le répertoire ``data/``.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"

# Ordre fixe (rapport : standard → contextual → copyright), pas le tri alphabétique.
_HARMBENCH_JSONL_FILES: tuple[str, ...] = (
    "harmbench_standard.jsonl",
    "harmbench_contextual.jsonl",
    "harmbench_copyright.jsonl",
)


def load_all_datasets() -> list[dict]:
    """
    Charge les trois jeux HarmBench (JSONL) depuis ``data/``.

    L'ordre est **fixe** : standard, contextual, copyright — reproductibilité
    des expériences et cohérence avec la méthodologie du rapport.

    Returns
    -------
    list[dict]
        Liste d'objets JSON (une ligne = un objet), dans l'ordre de lecture.
    """
    records: list[dict] = []
    for name in _HARMBENCH_JSONL_FILES:
        jsonl_path = DATA_DIR / name
        if not jsonl_path.is_file():
            continue
        with open(jsonl_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if line:
                    records.append(json.loads(line))
    return records
