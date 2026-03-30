"""
Chargement des jeux de prompts au format JSONL depuis le répertoire ``data/``.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"


def load_all_datasets() -> list[dict]:
    """
    Charge l'ensemble des fichiers ``*.jsonl`` présents dans ``data/``.

    Les fichiers sont parcourus par nom trié (ordre lexicographique) pour
    garantir un ordre d'échantillons déterministe entre exécutions.

    Returns
    -------
    list[dict]
        Liste d'objets JSON (une ligne = un objet), dans l'ordre de lecture.
    """
    records: list[dict] = []
    for jsonl_path in sorted(DATA_DIR.glob("*.jsonl")):
        with open(jsonl_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if line:
                    records.append(json.loads(line))
    return records
