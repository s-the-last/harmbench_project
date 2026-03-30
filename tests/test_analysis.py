import json
from pathlib import Path

import analysis as analysis_module


def test_load_results_document_envelope(tmp_path: Path):
    payload = {
        "experiment": {"model_name": "bloom"},
        "results": [{"prompt_id": 1, "scores": {"score_global": 0.5}}],
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    exp, rows = analysis_module.load_results_document(path)
    assert exp["model_name"] == "bloom"
    assert rows[0]["prompt_id"] == 1


def test_load_results_document_legacy_list(tmp_path: Path):
    payload = [{"type": "standard", "score_global": 0.2}]
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    exp, rows = analysis_module.load_results_document(path)
    assert exp is None
    assert rows[0]["type"] == "standard"
