"""
modules/llm/marlin_server/serve.py
-----------------------------------
FastAPI host for NemoStation/Marlin-2B (Phase 4 — remote backend).

Marlin-2B is a video-VLM that requires a GPU. SURVEILLANT runs on CPU,
so we run Marlin on a separate machine (Colab notebook, cloud VM,
workstation with NVIDIA GPU) and let the SURVEILLANT process talk to
it over HTTP.

Run on the GPU host:

    pip install fastapi uvicorn[standard] transformers>=5.7 torch \\
                torchcodec qwen-vl-utils pillow
    python -m surveillant.modules.llm.marlin_server.serve --port 8000

Then point SURVEILLANT at it by editing config/settings.py:

    DESCRIPTION_BACKEND = "marlin"
    MARLIN_HOST         = "http://<gpu-host>:8000"

The endpoint POST /describe expects:

    {
      "images_b64":    [<base64 png/jpg>, ...],
      "system_prompt": "<verbatim system prompt>",
      "user_prompt":   "<verbatim user prompt>",
      "expect_schema": [<field name>, ...],
      "color_palette": [<color name>, ...]
    }

and returns:

    {
      "attributes": {<canonical dict>},  # preferred
      "raw":        "<raw model output>" # for debugging
    }

If `attributes` is missing the client will fall back to JSON-cleaning `raw`.

This file intentionally has no project imports — it must run standalone
on a GPU host that doesn't have SURVEILLANT installed.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import time
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Lazy global model instance (loaded once on first request)
# ---------------------------------------------------------------------------

_model = None
_processor = None
_device = None


def _load_model_once():
    """Idempotent loader. Runs the heavy import + weight load only on first call."""
    global _model, _processor, _device
    if _model is not None:
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    if _device == "cpu":
        print("[MARLIN] WARNING — Marlin-2B requires a GPU. Loading on CPU will "
              "be very slow and may run out of RAM.")

    print(f"[MARLIN] loading NemoStation/Marlin-2B on {_device} ...")
    _model = AutoModelForCausalLM.from_pretrained(
        "NemoStation/Marlin-2B",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if _device == "cuda" else torch.float32,
        device_map={"": _device},
    )
    _processor = AutoProcessor.from_pretrained(
        "NemoStation/Marlin-2B", trust_remote_code=True,
    )
    print("[MARLIN] model ready.")


# ---------------------------------------------------------------------------
# JSON cleaning (mirrors describer._clean_json)
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL)

def _clean_json(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    text = raw.strip()
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    m = _FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except (ValueError, TypeError):
            pass
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e > s:
        try:
            return json.loads(text[s : e + 1])
        except (ValueError, TypeError):
            return None
    return None


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def _describe_images(
    images_b64:     List[str],
    system_prompt:  str,
    user_prompt:    str,
) -> Dict[str, Any]:
    """
    Pass the snapshots to Marlin-2B as a 1- or N-frame "video" using the
    Qwen-VL chat template. Marlin's visual tower is video-shaped, so we
    feed images as single-frame clips.
    """
    _load_model_once()

    from PIL import Image
    import torch

    images = []
    for b64 in images_b64:
        try:
            buf = base64.b64decode(b64)
            images.append(Image.open(io.BytesIO(buf)).convert("RGB"))
        except Exception as exc:
            print(f"[MARLIN] image decode failed: {exc}")

    if not images:
        return {"attributes": None, "raw": "", "error": "no decodable images"}

    # Build the chat with image inputs. Marlin extends Qwen-VL's template,
    # so the {"type": "image", "image": <PIL>} form is accepted.
    user_content = [{"type": "image", "image": im} for im in images]
    user_content.append({"type": "text", "text": user_prompt})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user",   "content": user_content},
    ]

    text = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = _processor(
        text=[text], images=images, return_tensors="pt", padding=True,
    ).to(_device)

    with torch.no_grad():
        out_ids = _model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.1,
        )
    raw = _processor.batch_decode(
        out_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )[0]

    parsed = _clean_json(raw)
    return {"attributes": parsed, "raw": raw}


# ---------------------------------------------------------------------------
# HTTP wiring
# ---------------------------------------------------------------------------

def build_app():
    """FastAPI app factory."""
    from fastapi import FastAPI

    app = FastAPI(title="SURVEILLANT Marlin-2B host")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status":      "ok",
            "model":       "NemoStation/Marlin-2B",
            "model_loaded": _model is not None,
            "device":      _device or "uninitialised",
        }

    @app.post("/describe")
    def describe(payload: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        result = _describe_images(
            images_b64    = payload.get("images_b64", []),
            system_prompt = payload.get("system_prompt", ""),
            user_prompt   = payload.get("user_prompt", ""),
        )
        result["elapsed_sec"] = round(time.time() - t0, 2)
        return result

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="SURVEILLANT Marlin-2B HTTP host")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--preload", action="store_true",
                        help="Load weights eagerly at startup instead of on first request.")
    args = parser.parse_args()

    if args.preload:
        _load_model_once()

    import uvicorn
    uvicorn.run(build_app(), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
