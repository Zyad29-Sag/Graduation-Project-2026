"""
modules/llm/describer.py
-------------------------
LLM-based body description for SURVEILLANT (Part 10 / Phase 4).

This module never raises into the embedding worker. Backend failures
return ``None`` so the DescriptionWorker can decide whether to retry.

Two backends behind one abstract interface, swappable via the
``DESCRIPTION_BACKEND`` config flag:

* ``QwenVLOllamaDescriber`` — local Ollama, CPU, default.
* ``MarlinRemoteDescriber`` — POSTs snapshots to a remote GPU host
  running ``modules/llm/marlin_server/serve.py``.

A third class (``QueryParser``) turns free-text operator queries into
the same JSON schema using a small text-only Ollama model — orders of
magnitude faster than the VLM and runs entirely on CPU.
"""

from __future__ import annotations

import base64
import json
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import (
    OLLAMA_HOST,
    OLLAMA_VLM_MODEL,
    OLLAMA_QUERY_MODEL,
    MARLIN_HOST,
    MARLIN_TIMEOUT_SEC,
    DESCRIPTION_BACKEND,
)


# ---------------------------------------------------------------------------
# Schema (kept in code so the prompts and SQL filters stay in sync)
# ---------------------------------------------------------------------------

REQUIRED_DESCRIPTION_FIELDS = (
    "gender", "age_range", "body_build", "height_class",
    "hair_color", "hair_length", "beard", "glasses",
    "headwear", "headwear_color",
    "clothing_top", "clothing_top_color",
    "clothing_bottom", "clothing_bottom_color",
    "accessories", "distinctive_features", "summary",
)

COLOR_PALETTE = (
    "red", "blue", "green", "yellow", "black", "white", "gray",
    "brown", "orange", "purple", "pink", "multi", "unknown",
)

# System prompt is identical across backends so model output stays comparable.
SYSTEM_PROMPT_DESCRIBE = (
    "You are a surveillance description assistant. Describe ONLY what you "
    "can see in the image. Do not infer beyond the image. Use the value "
    "'unknown' when you are not sure. Output ONLY a single JSON object — "
    "no commentary, no markdown fences, no prose."
)

USER_PROMPT_DESCRIBE = """\
Describe the person shown. Output a single JSON object with EXACTLY these fields:
{
  "gender":              "male" | "female" | "unknown",
  "age_range":           "child" | "teen" | "young_adult" | "adult" | "older_adult" | "unknown",
  "body_build":          "slim" | "average" | "heavy" | "unknown",
  "height_class":        "short" | "average" | "tall" | "unknown",
  "hair_color":          "black" | "brown" | "blonde" | "red" | "gray" | "white" | "unknown",
  "hair_length":         "short" | "medium" | "long" | "bald" | "unknown",
  "beard":               "yes" | "no" | "unknown",
  "glasses":             "yes" | "no" | "unknown",
  "headwear":            "<short phrase or 'none'>",
  "headwear_color":      "<color from palette or 'unknown'>",
  "clothing_top":        "<short phrase, e.g. 't-shirt', 'jacket', 'hoodie', 'shirt'>",
  "clothing_top_color":  "<color from palette>",
  "clothing_bottom":     "<short phrase, e.g. 'jeans', 'shorts', 'skirt', 'pants'>",
  "clothing_bottom_color": "<color from palette>",
  "accessories":         [<short phrases such as "backpack", "handbag", "umbrella", "watch">],
  "distinctive_features": "<one sentence or 'none'>",
  "summary":             "<one human-readable sentence>"
}
Palette = red | blue | green | yellow | black | white | gray | brown | orange | purple | pink | multi | unknown
"""

SYSTEM_PROMPT_PARSE_QUERY = (
    "You parse a security operator's spoken description of a person into a "
    "search filter for a database. Output ONLY a single JSON object — no "
    "commentary, no markdown fences. Omit any field the operator did not "
    "mention. Use 'no' for negations like 'without glasses' or 'no beard'."
)

USER_PROMPT_PARSE_QUERY = """\
Parse this query into the same JSON schema (omit unspecified fields):
{schema}

Query: {query}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL)


def _clean_json(raw: str) -> Optional[Dict[str, Any]]:
    """
    Robust JSON extraction from a possibly-noisy LLM output.

    Handles ```json fences, leading/trailing prose, and stray characters.
    Returns the parsed dict or None if the response is unrecoverable. The
    caller is expected to log the raw text on None so we can audit model
    failures later.
    """
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None

    # 1. Try the raw text directly.
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass

    # 2. Strip ```json ... ``` fences if present.
    m = _FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except (ValueError, TypeError):
            pass

    # 3. Last resort: locate the first '{' and the last '}' and try that slice.
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except (ValueError, TypeError):
            return None
    return None


def _coerce_to_schema(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalise an LLM response into our canonical schema.

    Missing fields become 'unknown' (or [] for accessories). Strings are
    lowercased. Colors outside the palette become 'unknown'. This keeps
    the search-side SQL filters predictable.
    """
    out: Dict[str, Any] = {}
    for f in REQUIRED_DESCRIPTION_FIELDS:
        v = parsed.get(f)
        if f == "accessories":
            if isinstance(v, list):
                out[f] = [str(x).strip().lower() for x in v if x]
            elif isinstance(v, str) and v.strip():
                out[f] = [v.strip().lower()]
            else:
                out[f] = []
        else:
            out[f] = (str(v).strip().lower() if v is not None and str(v).strip() else "unknown")

    # Palette guard — anything outside becomes 'unknown' so SQL filters
    # don't return surprises.
    for color_field in ("clothing_top_color", "clothing_bottom_color", "headwear_color"):
        if out[color_field] not in COLOR_PALETTE:
            out[color_field] = "unknown"

    # Summary should remain human-readable: preserve original case.
    if isinstance(parsed.get("summary"), str) and parsed["summary"].strip():
        out["summary"] = parsed["summary"].strip()
    return out


def _read_image_b64(path: str) -> Optional[str]:
    """Read an image file and return base64-encoded bytes (or None on failure)."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except OSError as exc:
        print(f"[DESCRIBE] read fail {path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Describer(ABC):
    """Abstract description backend. Implementations must never raise upward."""

    backend_name: str = "abstract"
    model_id:     str = ""

    @abstractmethod
    def describe(self, snapshot_paths: List[str]) -> Optional[Dict[str, Any]]:
        """
        Return a canonical-schema dict (see REQUIRED_DESCRIPTION_FIELDS),
        or None on any failure. The caller logs and decides whether to retry.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Qwen2.5-VL via local Ollama (default CPU backend)
# ---------------------------------------------------------------------------

class QwenVLOllamaDescriber(Describer):
    """
    Calls a local Ollama server running a vision LLM (default qwen2.5vl:2b).
    Talks via Ollama's HTTP /api/chat with image attachments.

    Run on the local machine; CPU-only is fine but slow (~5–15 s per image
    for qwen2.5vl:2b). Pull the model once with: ``ollama pull qwen2.5vl:2b``.
    """

    backend_name = "qwen-vl"

    def __init__(
        self,
        host: str = OLLAMA_HOST,
        model: str = OLLAMA_VLM_MODEL,
        timeout_sec: int = 120,
    ) -> None:
        self.host = host.rstrip("/")
        self.model_id = model
        self.timeout_sec = timeout_sec

    def describe(self, snapshot_paths: List[str]) -> Optional[Dict[str, Any]]:
        if not snapshot_paths:
            return None
        # 4A: single-image input. Multi-image consensus is deferred to 4C.
        path = snapshot_paths[0]
        img_b64 = _read_image_b64(path)
        if img_b64 is None:
            return None

        try:
            import requests  # local import so the dependency is optional
        except ImportError:
            print("[DESCRIBE] 'requests' not installed; cannot call Ollama.")
            return None

        payload = {
            "model":  self.model_id,
            "stream": False,
            "format": "json",  # Ollama's structured-output mode
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_DESCRIBE},
                {
                    "role":    "user",
                    "content": USER_PROMPT_DESCRIBE,
                    "images":  [img_b64],
                },
            ],
            "options": {
                "temperature": 0.1,
                "num_ctx":     2048,
            },
        }

        try:
            r = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=self.timeout_sec,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            print(f"[DESCRIBE] qwen-vl HTTP failure: {exc}")
            return None

        raw = (data.get("message") or {}).get("content") or ""
        parsed = _clean_json(raw)
        if parsed is None:
            preview = raw[:300].replace("\n", " ")
            print(f"[DESCRIBE] qwen-vl JSON parse failed; raw='{preview}...'")
            return None
        return _coerce_to_schema(parsed)


# ---------------------------------------------------------------------------
# Marlin-2B via remote GPU host
# ---------------------------------------------------------------------------

class MarlinRemoteDescriber(Describer):
    """
    POSTs snapshots to a FastAPI host running modules/llm/marlin_server/serve.py.

    Marlin-2B is a video-VLM that requires a GPU; SURVEILLANT runs on CPU.
    The remote host runs the model and exposes a single endpoint, so the
    CPU-only client only needs `requests`. If MARLIN_HOST is empty or the
    host is unreachable, describe() returns None.
    """

    backend_name = "marlin"

    def __init__(
        self,
        host: str = MARLIN_HOST,
        model_id: str = "NemoStation/Marlin-2B",
        timeout_sec: int = MARLIN_TIMEOUT_SEC,
    ) -> None:
        self.host = (host or "").rstrip("/")
        self.model_id = model_id
        self.timeout_sec = timeout_sec

    def describe(self, snapshot_paths: List[str]) -> Optional[Dict[str, Any]]:
        if not self.host:
            print("[DESCRIBE] marlin backend selected but MARLIN_HOST is empty.")
            return None
        if not snapshot_paths:
            return None

        try:
            import requests
        except ImportError:
            print("[DESCRIBE] 'requests' not installed; cannot call Marlin.")
            return None

        images_b64 = [b for b in (_read_image_b64(p) for p in snapshot_paths) if b]
        if not images_b64:
            return None

        payload = {
            "images_b64":      images_b64,
            "system_prompt":   SYSTEM_PROMPT_DESCRIBE,
            "user_prompt":     USER_PROMPT_DESCRIBE,
            "expect_schema":   list(REQUIRED_DESCRIPTION_FIELDS),
            "color_palette":   list(COLOR_PALETTE),
        }

        try:
            r = requests.post(
                f"{self.host}/describe",
                json=payload,
                timeout=self.timeout_sec,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            print(f"[DESCRIBE] marlin HTTP failure: {exc}")
            return None

        # The Marlin server returns either a structured dict directly OR a
        # 'raw' field that we still need to clean (depending on model output).
        if "attributes" in data and isinstance(data["attributes"], dict):
            return _coerce_to_schema(data["attributes"])

        parsed = _clean_json(data.get("raw", ""))
        if parsed is None:
            return None
        return _coerce_to_schema(parsed)


# ---------------------------------------------------------------------------
# Query parser (text-only Ollama)
# ---------------------------------------------------------------------------

class QueryParser:
    """
    Parses a free-text operator query into a partial description filter.

    Uses a small text-only Ollama model (default qwen2.5:3b). Missing
    fields = no filter on that field. Returns the dict directly so the
    TextSearchEngine can pass it straight to search_persons_by_attributes.
    """

    def __init__(
        self,
        host: str = OLLAMA_HOST,
        model: str = OLLAMA_QUERY_MODEL,
        timeout_sec: int = 30,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout_sec = timeout_sec

    def parse(self, query: str) -> Dict[str, Any]:
        try:
            import requests
        except ImportError:
            print("[QUERY] 'requests' not installed; cannot call Ollama.")
            return self._rule_based_fallback(query)

        # Compact schema string for the prompt.
        schema = "{\n" + ",\n".join(f'  "{f}": ...' for f in REQUIRED_DESCRIPTION_FIELDS) + "\n}"
        payload = {
            "model":  self.model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_PARSE_QUERY},
                {"role": "user",   "content": USER_PROMPT_PARSE_QUERY.format(
                    schema=schema, query=query,
                )},
            ],
            "options": {"temperature": 0.0, "num_ctx": 2048},
        }

        try:
            r = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=self.timeout_sec,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            print(f"[QUERY] LLM failure: {exc} — falling back to rule-based parser.")
            return self._rule_based_fallback(query)

        raw = (data.get("message") or {}).get("content") or ""
        parsed = _clean_json(raw)
        if not parsed:
            print(f"[QUERY] LLM returned unparseable JSON — rule-based fallback.")
            return self._rule_based_fallback(query)

        # Don't coerce to full schema here — missing fields must STAY missing
        # so the SQL builder ignores them. Just lowercase known string values.
        cleaned: Dict[str, Any] = {}
        for f in REQUIRED_DESCRIPTION_FIELDS:
            if f not in parsed:
                continue
            v = parsed[f]
            if f == "accessories":
                if isinstance(v, list):
                    cleaned[f] = [str(x).strip().lower() for x in v if x]
                elif isinstance(v, str) and v.strip():
                    cleaned[f] = [v.strip().lower()]
            elif isinstance(v, str) and v.strip():
                cleaned[f] = v.strip().lower()
            elif v is not None:
                cleaned[f] = v
        return cleaned

    # --- Rule-based fallback ----------------------------------------------

    _KEYWORDS = {
        # gender
        "male":   ("man", "guy", "male", "boy", "gentleman"),
        "female": ("woman", "lady", "female", "girl"),
    }
    _COLORS = COLOR_PALETTE
    _GARMENTS_TOP = ("t-shirt", "tshirt", "shirt", "jacket", "hoodie", "coat", "sweater", "blouse", "vest")
    _GARMENTS_BOTTOM = ("jeans", "pants", "shorts", "skirt", "trousers", "dress")
    _HEADWEAR = ("hat", "cap", "beanie", "hood", "scarf", "helmet")
    _ACCESSORIES = ("backpack", "handbag", "bag", "umbrella", "watch", "phone")

    def _rule_based_fallback(self, query: str) -> Dict[str, Any]:
        q = query.lower()
        out: Dict[str, Any] = {}

        # Gender — check female FIRST and use word boundaries so "woman"
        # doesn't trigger the "man" substring match. (This is the kind of
        # bug a real LLM parser doesn't have.)
        if any(re.search(rf"\b{kw}\b", q) for kw in self._KEYWORDS["female"]):
            out["gender"] = "female"
        elif any(re.search(rf"\b{kw}\b", q) for kw in self._KEYWORDS["male"]):
            out["gender"] = "male"

        # Build (word-boundary so "chubby" doesn't match inside other words)
        if re.search(r"\b(fat|heavy|overweight|chubby|stocky)\b", q):
            out["body_build"] = "heavy"
        elif re.search(r"\b(slim|thin|skinny|lean)\b", q):
            out["body_build"] = "slim"

        # Negation for binary fields
        if re.search(r"\b(no|without)\s+glasses\b", q): out["glasses"] = "no"
        elif "glasses" in q:                            out["glasses"] = "yes"
        if re.search(r"\b(no|without)\s+beard\b", q):   out["beard"] = "no"
        elif "beard" in q:                              out["beard"] = "yes"

        # Top + top color: "<color> <top-garment>"
        for color in self._COLORS:
            for top in self._GARMENTS_TOP:
                if re.search(rf"\b{color}\b.{{0,15}}\b{top}\b", q):
                    out["clothing_top"]       = top
                    out["clothing_top_color"] = color
                    break
        # Bottom
        for color in self._COLORS:
            for bot in self._GARMENTS_BOTTOM:
                if re.search(rf"\b{color}\b.{{0,15}}\b{bot}\b", q):
                    out["clothing_bottom"]       = bot
                    out["clothing_bottom_color"] = color
                    break
        # Headwear
        for color in self._COLORS:
            for hw in self._HEADWEAR:
                if re.search(rf"\b{color}\b.{{0,15}}\b{hw}\b", q):
                    out["headwear"]       = hw
                    out["headwear_color"] = color
                    break

        # Accessories
        accs = [a for a in self._ACCESSORIES if re.search(rf"\b{a}\b", q)]
        if accs:
            out["accessories"] = accs

        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_describer(backend: Optional[str] = None) -> Describer:
    """
    Construct the configured describer backend. Falls back to qwen-vl with
    a warning if Marlin is requested but MARLIN_HOST is empty.
    """
    name = (backend or DESCRIPTION_BACKEND or "qwen-vl").lower()
    if name == "marlin":
        if MARLIN_HOST:
            return MarlinRemoteDescriber()
        print(
            "[DESCRIBE] DESCRIPTION_BACKEND='marlin' but MARLIN_HOST is empty — "
            "falling back to local qwen-vl. Set MARLIN_HOST in settings.py to "
            "use the GPU host."
        )
    return QwenVLOllamaDescriber()
