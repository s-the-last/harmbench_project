"""
Agrégation des résultats d'évaluation : moyennes par modèle et par type de test,
synthèse des vulnérabilités, export CSV optionnel.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _configure_stdout_utf8() -> None:
    """Évite les erreurs d'encodage des caractères accentués sous Windows (cp1252)."""
    out = sys.stdout
    if hasattr(out, "reconfigure"):
        try:
            out.reconfigure(encoding="utf-8")
        except Exception:
            pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EXAMPLES_PER_TYPE = 3


def load_results_document(path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """
    Charge un fichier ``results.json`` au format enveloppe ou ancien format (liste seule).

    Returns
    -------
    experiment, rows
        ``experiment`` est ``None`` si le fichier est une liste brute.
    """
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "results" in payload:
        return payload.get("experiment"), list(payload["results"])
    if isinstance(payload, list):
        return None, payload
    raise ValueError(f"Format JSON non reconnu : {path}")


def _row_score_global(row: dict[str, Any]) -> float:
    """Extrait ``score_global`` depuis une ligne au format neuf (scores imbriqués) ou ancien."""
    scores = row.get("scores")
    if isinstance(scores, dict) and "score_global" in scores:
        return float(scores["score_global"])
    return float(row.get("score_global", 0.0))


def _row_metric(row: dict[str, Any], key: str) -> float:
    """Lit un indicateur binaire dans ``scores`` ou à la racine (rétrocompatibilité)."""
    scores = row.get("scores")
    if isinstance(scores, dict) and key in scores:
        return float(scores[key])
    return float(row.get(key, 0))


def _prompt_type(row: dict[str, Any]) -> str:
    return str(row.get("prompt_type", row.get("type", "")))


def _model_name_from_row(row: dict[str, Any], experiment: dict[str, Any] | None) -> str:
    name = row.get("model_name") or row.get("model")
    if name:
        return str(name)
    if experiment and experiment.get("model_name"):
        return str(experiment["model_name"])
    return "unknown"


def summarize_file(path: Path) -> None:
    """Affiche statistiques détaillées pour un fichier ``results.json``."""
    experiment, rows = load_results_document(path)
    print(f"\n=== Fichier : {path} ===")
    if experiment:
        print(f"Modèle (métadonnées experiment) : {experiment.get('model_name')}")
        print(
            f"Période UTC : {experiment.get('started_at_utc')} → {experiment.get('completed_at_utc')}"
        )
        if experiment.get("dry_run"):
            print("Mode : dry-run")

    by_type: dict[str, dict[str, float]] = defaultdict(
        lambda: {"n": 0, "score_sum": 0.0, "refusal_sum": 0.0, "harmful_sum": 0.0, "nuisance_sum": 0.0}
    )
    for row in rows:
        t = _prompt_type(row)
        agg = by_type[t]
        agg["n"] += 1
        agg["score_sum"] += _row_score_global(row)
        agg["refusal_sum"] += _row_metric(row, "refusal")
        agg["harmful_sum"] += _row_metric(row, "harmful")
        agg["nuisance_sum"] += _row_metric(row, "nuisance")

    for t in sorted(by_type.keys()):
        agg = by_type[t]
        n = max(int(agg["n"]), 1)
        print(f"\n  Type [{t}] (n={int(agg['n'])})")
        print(f"    Score moyen (score_global) : {agg['score_sum'] / n:.3f}")
        print(f"    Taux de refus lexical      : {agg['refusal_sum'] / n:.3f}")
        print(f"    Taux de complaisance       : {agg['harmful_sum'] / n:.3f}")
        print(f"    Taux de nuisance           : {agg['nuisance_sum'] / n:.3f}")

        shown = 0
        for row in rows:
            if _prompt_type(row) != t:
                continue
            if shown >= EXAMPLES_PER_TYPE:
                break
            raw = row.get("raw_response", row.get("response", ""))
            preview = raw[:100] + ("..." if len(raw) > 100 else "")
            print(f"    Exemple prompt : {row.get('prompt_text', row.get('prompt', ''))}")
            print(f"    Extrait réponse : {preview}")
            print(f"    score_global : {_row_score_global(row):.3f}")
            shown += 1


def cross_model_summary(result_paths: list[Path]) -> None:
    """
    Calcule la moyenne de ``score_global`` par modèle et par type sur l'ensemble des fichiers.
    """
    model_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0, "score_sum": 0.0}
    )
    type_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0, "score_sum": 0.0}
    )
    type_by_model: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"count": 0, "score_sum": 0.0}
    )

    for path in result_paths:
        experiment, rows = load_results_document(path)
        for row in rows:
            model = _model_name_from_row(row, experiment)
            ptype = _prompt_type(row)
            sg = _row_score_global(row)

            model_totals[model]["count"] += 1
            model_totals[model]["score_sum"] += sg

            type_totals[ptype]["count"] += 1
            type_totals[ptype]["score_sum"] += sg

            key = (model, ptype)
            type_by_model[key]["count"] += 1
            type_by_model[key]["score_sum"] += sg

    print("\n--- Moyennes de score_global par modèle (tous types confondus) ---")
    for model in sorted(model_totals.keys()):
        m = model_totals[model]
        n = max(int(m["count"]), 1)
        print(f"  {model}: {m['score_sum'] / n:.3f} (n={int(m['count'])})")

    print("\n--- Moyennes de score_global par type de test (tous modèles confondus) ---")
    for ptype in sorted(type_totals.keys()):
        m = type_totals[ptype]
        n = max(int(m["count"]), 1)
        print(f"  {ptype}: {m['score_sum'] / n:.3f} (n={int(m['count'])})")

    vulnerability_commentary(result_paths, type_by_model, type_totals)


def vulnerability_commentary(
    result_paths: list[Path],
    type_by_model: dict[tuple[str, str], dict[str, float]],
    type_totals: dict[str, dict[str, float]],
) -> None:
    """
    Met en évidence des écarts entre catégories (ex. contextual vs standard).
    """
    print("\n--- Synthèse des vulnérabilités (heuristique) ---")

    def avg_for(ptype: str) -> float | None:
        m = type_totals.get(ptype)
        if not m or m["count"] == 0:
            return None
        return m["score_sum"] / m["count"]

    std = avg_for("standard")
    ctx = avg_for("contextual")
    cpy = avg_for("copyright")

    if std is not None and ctx is not None:
        gap = std - ctx
        if gap > 0.05:
            print(
                f"  - Écart standard - contextual = {gap:+.3f} : le modèle semble "
                "plus vulnérable aux attaques par injection de contexte "
                "(scores plus bas sur [contextual])."
            )
        elif gap < -0.05:
            print(
                f"  - Écart standard - contextual = {gap:+.3f} : les prompts contextuels obtiennent "
                "des scores plus élevés que le standard (interprétation dépend du jeu de données)."
            )
        else:
            print(
                f"  - Écart standard - contextual = {gap:+.3f} : différence modérée entre les deux dimensions."
            )

    if std is not None and cpy is not None:
        gap2 = std - cpy
        print(
            f"  - Écart standard - copyright = {gap2:+.3f} "
            "(sensibilité relative aux requêtes copyright)."
        )

    if len(result_paths) < 2:
        print(
            "  - Pour comparer deux modèles sur les mêmes types, exécutez evaluate.py pour chaque modèle "
            "puis relancez cette analyse."
        )


def export_results_csv(result_paths: list[Path], csv_path: Path) -> None:
    """Aplatit les résultats de plusieurs fichiers JSON en un seul CSV."""
    fieldnames = [
        "source_file",
        "model_name",
        "prompt_id",
        "prompt_type",
        "prompt_category",
        "expected_behavior",
        "timestamp_utc",
        "prompt_text",
        "raw_response",
        "score_global",
        "refusal",
        "justification",
        "politeness",
        "toxicity",
        "harmful",
        "nuisance",
        "helpfulness",
        "inappropriate_refusal",
        "dry_run",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for path in result_paths:
            experiment, rows = load_results_document(path)
            for row in rows:
                scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
                writer.writerow(
                    {
                        "source_file": path.name,
                        "model_name": _model_name_from_row(row, experiment),
                        "prompt_id": row.get("prompt_id", row.get("id", "")),
                        "prompt_type": _prompt_type(row),
                        "prompt_category": row.get("prompt_category", row.get("category", "")),
                        "expected_behavior": row.get(
                            "expected_behavior", row.get("expected", "")
                        ),
                        "timestamp_utc": row.get("timestamp_utc", ""),
                        "prompt_text": row.get("prompt_text", row.get("prompt", "")),
                        "raw_response": row.get("raw_response", row.get("response", "")),
                        "score_global": scores.get("score_global", row.get("score_global", "")),
                        "refusal": scores.get("refusal", row.get("refusal", "")),
                        "justification": scores.get(
                            "justification", row.get("justification", "")
                        ),
                        "politeness": scores.get("politeness", row.get("politeness", "")),
                        "toxicity": scores.get("toxicity", row.get("toxicity", "")),
                        "harmful": scores.get("harmful", row.get("harmful", "")),
                        "nuisance": scores.get("nuisance", row.get("nuisance", "")),
                        "helpfulness": scores.get("helpfulness", row.get("helpfulness", "")),
                        "inappropriate_refusal": scores.get(
                            "inappropriate_refusal",
                            row.get("inappropriate_refusal", ""),
                        ),
                        "dry_run": row.get(
                            "dry_run",
                            (experiment or {}).get("dry_run", ""),
                        ),
                    }
                )
    print(f"Export CSV écrit : {csv_path}")


def main() -> None:
    _configure_stdout_utf8()
    parser = argparse.ArgumentParser(
        description="Analyse les résultats dans outputs/*/results.json"
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default=None,
        metavar="PATH",
        help="Chemin du fichier CSV agrégé (ex. outputs/aggregate_results.csv).",
    )
    args = parser.parse_args()

    result_paths = sorted(OUTPUTS_DIR.glob("*/results.json"))
    if not result_paths:
        print(f"Aucun results.json sous {OUTPUTS_DIR}. Exécutez d'abord src/evaluate.py.")
        return

    for path in result_paths:
        summarize_file(path)

    cross_model_summary(result_paths)

    if args.export_csv:
        export_results_csv(result_paths, Path(args.export_csv))


if __name__ == "__main__":
    main()
