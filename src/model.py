"""
Appels à l'API Hugging Face Inference pour la génération de texte.

Les URL et identifiants de modèles sont centralisés dans ``MODELS``.
"""

from __future__ import annotations

import os
import time

import requests

HF_TOKEN: str = os.getenv("HF_TOKEN", "")
HEADERS: dict[str, str] = (
    {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
)

MODELS: dict[str, str] = {
    "bloom": "https://api-inference.huggingface.co/models/bigscience/bloom-560m",
    "flan": "https://api-inference.huggingface.co/models/google/flan-t5-base",
}

MAX_RETRIES: int = 5
RETRY_DELAY_SEC: float = 10.0


def _extract_text(payload: object) -> str:
    """
    Extrait la chaîne générée à partir du corps JSON renvoyé par l'API Inference.

    Gère les formes courantes : liste de dicts avec ``generated_text`` ou
    ``summary_text``, ou dict avec erreur / ``generated_text``.
    """
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict):
            if "generated_text" in first:
                return str(first["generated_text"])
            if "summary_text" in first:
                return str(first["summary_text"])
    if isinstance(payload, dict):
        if "generated_text" in payload:
            return str(payload["generated_text"])
        if "error" in payload:
            return f"ERROR: {payload.get('error', payload)}"
    return "ERROR"


def _should_retry(response: requests.Response, body: object) -> bool:
    """Indique si une nouvelle tentative est pertinente (saturation, chargement du modèle, etc.)."""
    if response.status_code == 503:
        return True
    if isinstance(body, dict) and body.get("error"):
        err = str(body["error"]).lower()
        if "loading" in err or "unavailable" in err or "warm" in err:
            return True
    return False


def query_model(
    prompt: str,
    model_name: str = "bloom",
    max_retries: int = MAX_RETRIES,
) -> str:
    """
    Envoie ``prompt`` au modèle ``model_name`` via l'API Inference.

    Parameters
    ----------
    prompt
        Texte utilisateur à compléter ou à traiter selon le modèle.
    model_name
        Clé parmi ``MODELS`` (ex. ``"bloom"``, ``"flan"``).
    max_retries
        Nombre maximal de tentatives en cas d'erreur transitoire.

    Returns
    -------
    str
        Texte généré ou une chaîne commençant par ``ERROR`` en cas d'échec.
    """
    if model_name not in MODELS:
        return f"ERROR: unknown model {model_name!r}"

    api_url = MODELS[model_name]
    request_body = {"inputs": prompt, "parameters": {"max_new_tokens": 120}}

    for attempt in range(max_retries):
        response = requests.post(
            api_url, headers=HEADERS, json=request_body, timeout=120
        )
        try:
            parsed_body = response.json()
        except Exception:
            if attempt < max_retries - 1 and response.status_code >= 500:
                time.sleep(RETRY_DELAY_SEC)
                continue
            return "ERROR"

        if _should_retry(response, parsed_body) and attempt < max_retries - 1:
            time.sleep(RETRY_DELAY_SEC)
            continue

        text = _extract_text(parsed_body)
        if text == "ERROR" and response.status_code >= 400 and attempt < max_retries - 1:
            time.sleep(RETRY_DELAY_SEC)
            continue
        return text

    return "ERROR"
