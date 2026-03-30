"""
Script principal d'évaluation : charge les prompts, interroge un modèle, score les réponses.

Les sorties sont structurées en JSON avec un bloc ``experiment`` (traçabilité)
et une liste ``results`` (une entrée par prompt).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from loader import load_all_datasets  # noqa: E402
from model import HF_TOKEN, query_model  # noqa: E402
from scorer import score_response, scorer_metadata  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def dry_run_response(expected_behavior: str) -> str:
    """
    Produit une réponse synthétique pour valider le pipeline sans appel réseau.

    Parameters
    ----------
    expected_behavior
        ``"refuse"`` ou ``"answer"``, aligné sur le champ ``expected`` des JSONL.
    """
    if expected_behavior == "refuse":
        return (
            "I'm sorry, I cannot assist with harmful or illegal requests "
            "because that would be unsafe."
        )
    return (
        "This is a synthetic dry-run answer for testing. "
        "Photosynthesis is how plants turn light into chemical energy. " * 2
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Évalue un modèle (Hugging Face Inference) sur les prompts JSONL "
            "du répertoire data/."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bloom",
        help="Identifiant local du modèle : bloom ou flan.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="N'appelle pas l'API ; réponses factices pour tests de chaîne.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Traiter uniquement les N premiers enregistrements après chargement.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Délai en secondes entre deux appels API (hors dry-run).",
    )
    args = parser.parse_args()

    if not args.dry_run and not HF_TOKEN:
        print(
            "Attention : HF_TOKEN n'est pas défini. L'API peut refuser les "
            "requêtes. Utilisez --dry-run ou définissez HF_TOKEN.\n"
        )

    output_dir = PROJECT_ROOT / "outputs" / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_records = load_all_datasets()
    if args.limit is not None:
        prompt_records = prompt_records[: args.limit]

    started_at = datetime.now(timezone.utc)
    result_rows: list[dict] = []

    for record in prompt_records:
        prompt_id = record["id"]
        prompt_type = record["type"]
        print(f"\nTesting prompt_id={prompt_id} (type={prompt_type})")

        if args.dry_run:
            raw_response = dry_run_response(record["expected"])
        else:
            raw_response = query_model(record["prompt"], model_name=args.model)

        score_dict = score_response(raw_response, record["expected"])
        observed_at = datetime.now(timezone.utc)

        row: dict = {
            "prompt_id": prompt_id,
            "prompt_type": prompt_type,
            "prompt_category": record["category"],
            "expected_behavior": record["expected"],
            "prompt_text": record["prompt"],
            "model_name": args.model,
            "timestamp_utc": observed_at.isoformat().replace("+00:00", "Z"),
            "raw_response": raw_response,
            "scores": score_dict,
        }
        if args.dry_run:
            row["dry_run"] = True
        result_rows.append(row)

        if not args.dry_run:
            time.sleep(args.sleep)

    completed_at = datetime.now(timezone.utc)
    payload = {
        "experiment": {
            "model_name": args.model,
            "started_at_utc": started_at.isoformat().replace("+00:00", "Z"),
            "completed_at_utc": completed_at.isoformat().replace("+00:00", "Z"),
            "dry_run": bool(args.dry_run),
            "limit": args.limit,
            "sleep_seconds": args.sleep,
            "scorer": scorer_metadata(),
        },
        "results": result_rows,
    }

    output_path = output_dir / "results.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
