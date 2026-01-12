# llm_agent.py
from __future__ import annotations
import json
import os
from typing import List, Optional, Any, Dict
import requests
import os
import requests
from plan import Plan

class LLMPlannerError(Exception):
    pass

def _openrouter_request(messages, model):

    # Send messages to OpenRouter and return the model response text

    # Get API key from environment
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise LLMPlannerError("OPENROUTER_API_KEY is not set")

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
    }

    response = requests.post(url, headers=headers, json=payload)

    # Check if request failed
    if response.status_code != 200:
        raise LLMPlannerError(
            f"OpenRouter request failed with status {response.status_code}: {response.text}"
        )

    data = response.json()

    # Extract model output
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise LLMPlannerError("Could not parse response from OpenRouter")


def _parse_json(text: str):
    raw = text
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise LLMPlannerError(f"LLM did not return valid JSON: {e}\nRaw:\n{text}")


def _resolve_columns(cols: Optional[list[Any]], available_columns: List[str]):
    if cols is None:
        return None
    if not isinstance(cols, list):
        return None

    col_map = {c.lower(): c for c in available_columns}
    resolved: list[str] = []
    for c in cols:
        if isinstance(c, str) and c.lower() in col_map:
            resolved.append(col_map[c.lower()])

    out: list[str] = []
    seen = set()
    for c in resolved:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out if out else None


def make_plan_llm(instruction: str, available_columns: List[str]):

    # Uses an LLM to convert instruction to JSON plan

    model_name = os.getenv("LLM_MODEL", "openai/gpt-4o-mini").strip()

    system = (
        "Convert the user instruction into a JSON object matching the schema below. "
        "Return only JSON, no explanations.\n\n"
        "Schema:\n"
        "{\n"
        '  "model": "logistic_regression" | "random_forest",\n'
        '  "drop_columns": [string, ...],\n'
        '  "keep_only_columns": [string, ...] | null\n'
        "}\n\n"
        "Rules:\n"
        "- Only use columns from available_columns.\n"
        "- If instruction says 'drop all columns except X', set keep_only_columns=[X].\n"
        "- If no model mentioned, use logistic_regression.\n"
        "- If no columns to drop, drop_columns=[].\n"
    )

    user = (
        f"available_columns: {available_columns}\n"
        f"instruction: {instruction}\n"
    )

    content = _openrouter_request(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model=model_name,
    )

    plan_dict = _parse_json(content)

    model = plan_dict.get("model", "logistic_regression")
    if model not in ("logistic_regression", "random_forest"):
        model = "logistic_regression"

    drop_cols = plan_dict.get("drop_columns", [])
    if not isinstance(drop_cols, list):
        drop_cols = []

    keep_cols = plan_dict.get("keep_only_columns", None)

    resolved_drop = _resolve_columns(drop_cols, available_columns) or []
    resolved_keep = _resolve_columns(keep_cols, available_columns)

    return Plan(
        model=model,
        drop_columns=resolved_drop,
        keep_only_columns=resolved_keep,
    )
