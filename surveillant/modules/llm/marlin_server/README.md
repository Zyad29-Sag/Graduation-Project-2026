# SURVEILLANT — Marlin-2B Remote Host

[Marlin-2B](https://huggingface.co/NemoStation/Marlin-2B) is a 2B-parameter video VLM (Apache-2.0). Its visual tower is shaped for video clips, so we feed it the snapshot crops as 1-frame clips when generating a description.

**Marlin-2B requires a GPU.** SURVEILLANT's main pipeline runs on CPU, so we host Marlin on a separate machine and call it over HTTP via the `MarlinRemoteDescriber` backend.

If you don't have GPU access, set `DESCRIPTION_BACKEND = "qwen-vl"` in `config/settings.py` and use the local Ollama backend instead. It runs on CPU and produces the same JSON schema.

---

## Deployment options

### Option A — Google Colab notebook (free tier T4 GPU)

```python
# Cell 1: install
!pip install -q fastapi "uvicorn[standard]" "transformers>=5.7" \
                torchcodec qwen-vl-utils pillow pyngrok

# Cell 2: download the SURVEILLANT serve.py
!curl -sLO https://raw.githubusercontent.com/<YOUR_REPO>/main/surveillant/modules/llm/marlin_server/serve.py

# Cell 3: run the server in the background + expose via ngrok
import subprocess, time
subprocess.Popen(["python", "serve.py", "--port", "8000", "--preload"])
time.sleep(60)  # wait for the model to download + load

from pyngrok import ngrok
public = ngrok.connect(8000)
print(public.public_url)        # ← paste this into MARLIN_HOST on your laptop
```

Then on the laptop edit `config/settings.py`:

```python
DESCRIPTION_BACKEND = "marlin"
MARLIN_HOST         = "https://abcdef-ngrok-url"  # from Colab cell 3
```

Run `python main.py --phase 2 ...` as usual.

### Option B — Cloud VM (AWS / Hetzner / RunPod with an NVIDIA GPU)

```bash
# Install dependencies
pip install fastapi "uvicorn[standard]" "transformers>=5.7" \
            torchcodec qwen-vl-utils pillow torch

# Start the host (preload to fail fast if GPU isn't visible)
python -m surveillant.modules.llm.marlin_server.serve \
    --host 0.0.0.0 --port 8000 --preload
```

Open port 8000 in the security group and point `MARLIN_HOST` at the public IP.

### Option C — Local workstation with a discrete GPU

Same as Option B but the URL is `http://localhost:8000` (or LAN IP if SURVEILLANT runs on a different machine).

---

## API

### `GET /health`

Returns model load status:

```json
{
  "status": "ok",
  "model": "NemoStation/Marlin-2B",
  "model_loaded": true,
  "device": "cuda"
}
```

### `POST /describe`

Request:

```json
{
  "images_b64":    ["<base64-jpeg>", ...],
  "system_prompt": "...",
  "user_prompt":   "...",
  "expect_schema": ["gender", "age_range", ...],
  "color_palette": ["red", "blue", ...]
}
```

Response:

```json
{
  "attributes": { "gender": "male", "clothing_top": "t-shirt", ... },
  "raw":        "<raw model text>",
  "elapsed_sec": 4.2
}
```

If `attributes` is missing or null, the SURVEILLANT client falls back to JSON-cleaning the `raw` field.

---

## Performance notes

* Marlin-2B on a single T4 ≈ 4–10 s per description (1 image).
* First request after server start triggers the model download (≈ 4 GB) and weight load (≈ 30–60 s). Use `--preload` to hide the delay.
* For batched throughput, keep the SURVEILLANT-side queue depth low (≤ 5 in flight) — the HF wrapper here isn't optimised for concurrent inference; one request at a time is the safe default.
