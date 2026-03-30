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

# Clés courtes : trois familles alignées sur le rapport (API Inference uniforme).
# Descriptions pour traçabilité (experiment / documentation).
MODEL_DESCRIPTIONS: dict[str, str] = {
    "gpt2": "Famille GPT (causal LM, filtrage variable selon le déploiement)",
    "phi3_mini": "Modèle instruct aligné (proxy « conversationnel alternatif » / politique stricte)",
    "tinyllama": "Modèle open-source type Llama (paramètres publics, garde-fous souvent plus légers)",
}

MODELS: dict[str, str] = {
    "gpt2": "https://api-inference.huggingface.co/models/openai-community/gpt2",
    "phi3_mini": "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct",
    "tinyllama": "https://api-inference.huggingface.co/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
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
        if "summary_text" in payload:
            return str(payload["summary_text"])
        # Erreur API : parfois chaîne, parfois objet détaillé
        if "error" in payload:
            err = payload.get("error", payload)
            if isinstance(err, str):
                return f"ERROR: {err}"
            return f"ERROR: {err}"
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
    model_name: str = "gpt2",
    max_retries: int = MAX_RETRIES,
) -> str:
    """
    Envoie ``prompt`` au modèle ``model_name`` via l'API Inference.

    Parameters
    ----------
    prompt
        Texte utilisateur à compléter ou à traiter selon le modèle.
    model_name
        Clé parmi ``MODELS`` (ex. ``"gpt2"``, ``"phi3_mini"``, ``"tinyllama"``).
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
        # 401/403 : token manquant ou refus d'accès — inutile de boucler
        if response.status_code in (401, 403):
            return (
                f"ERROR: HTTP {response.status_code} — "
                "vérifiez HF_TOKEN (https://huggingface.co/settings/tokens)"
            )
        if text == "ERROR" and response.status_code >= 400 and attempt < max_retries - 1:
            time.sleep(RETRY_DELAY_SEC)
            continue
        return text

    return "ERROR"
