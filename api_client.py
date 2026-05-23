"""
Shared HTTP client for the deployed Anentra Library public REST API.

All demos access anentra_lib functions exclusively through this module.
The library itself is private — never import it directly in demo code.
"""

import os
import requests
from typing import Any, Dict, List

BASE_URL = os.environ.get(
    "ANENTRA_API_BASE_URL",
    "https://3ifcdwbj09.execute-api.us-east-1.amazonaws.com",
).rstrip("/")

_API_KEY = os.environ.get(
    "ANENTRA_API_KEY",
    "YiLXEF99WgaV-WGnR6HxWAMPtBoLOlpSrtsr5nt1EyoRsWQJb60lMA",
)

_HEADERS = {"x-api-key": _API_KEY, "Content-Type": "application/json"}


def _post(path: str, payload: Dict) -> Dict:
    resp = requests.post(
        f"{BASE_URL}{path}", json=payload, headers=_HEADERS, timeout=30
    )
    resp.raise_for_status()
    return resp.json()


def health() -> Dict:
    """GET /health — public endpoint, no API key required."""
    resp = requests.get(f"{BASE_URL}/health", timeout=10)
    resp.raise_for_status()
    return resp.json()


def remove_pii(text: str, hash_identifiers: bool = False) -> str:
    """POST /pii/remove — returns cleaned text with PII removed or hashed."""
    result = _post("/pii/remove", {"text": text, "hash_identifiers": hash_identifiers})
    return result["cleaned_text"]


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """POST /text/extract-json — extracts first JSON object embedded in text."""
    result = _post("/text/extract-json", {"text": text})
    return result["extracted"]


def calculate_metrics(true_labels: List[int], predictions: List[int]) -> Dict:
    """POST /metrics/classification — returns {precision, recall, num_samples}."""
    return _post(
        "/metrics/classification",
        {"true_labels": true_labels, "predictions": predictions},
    )


def build_recipe_prompt(
    dietary_restrictions: str,
    cuisine_preferences: str,
    ingredients_available: str,
) -> str:
    """POST /prompts/recipe — returns a structured recipe blog prompt string."""
    result = _post(
        "/prompts/recipe",
        {
            "dietary_restrictions": dietary_restrictions,
            "cuisine_preferences": cuisine_preferences,
            "ingredients_available": ingredients_available,
        },
    )
    return result["prompt"]


def check_syntax(source_code: str) -> Dict:
    """POST /code/check-syntax — returns {"is_valid": bool, "errors": list}."""
    return _post("/code/check-syntax", {"source_code": source_code})
