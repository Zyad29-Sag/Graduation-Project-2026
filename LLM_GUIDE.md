# SURVEILLANT — LLM Body Description & Search: Complete Setup Guide

This guide covers everything you need to get the LLM pipeline running, from installation to live search. There are **two independent paths** — pick the one that fits your hardware.

| Path | Hardware needed | Setup time | Description quality |
|---|---|---|---|
| **Path A — Local Ollama** | Any CPU laptop ✅ | ~10 min | Good (~5–15 s per person) |
| **Path B — Marlin-2B (remote GPU)** | GPU machine (Colab/cloud/desktop) | ~20 min | Better (video-VLM) |

Start with **Path A** if you're not sure. You can switch to Path B later with a single settings change.

---

## Path A — Local Ollama (Default, CPU-only)

### Step 1: Install Ollama

Go to **https://ollama.com/download** and install the version for your OS (Windows / Mac / Linux).

After installing, verify it's running:

```
ollama --version
```

You should see something like `ollama version 0.3.x`. If not, launch the Ollama app from your Start Menu / Applications folder first.

---

### Step 2: Pull the one model

SURVEILLANT uses a **single model** for both jobs — describing people *and*
parsing your text searches. You only need to download it once:

```bash
# `ollama run` downloads it (if missing) AND opens an interactive chat so
# you can eyeball the model once. Type a test line, then Ctrl-D to exit —
# the model stays cached for SURVEILLANT to use.
ollama run qwen2.5vl:3b
```

> **Note:** SURVEILLANT does not call `ollama run`/`ollama pull` itself — it
> talks to the Ollama **daemon** over HTTP and loads the model on demand. The
> command above just makes sure the model is downloaded and cached. `ollama
> run` doubles as a quick smoke test.
>
> The text-search query parser also uses `qwen2.5vl:3b`, but only as a
> fallback — an instant rule-based parser handles common queries (gender,
> build, garment+colour, glasses/beard, accessories, negation) with no model
> call at all, so most searches return immediately.

**Verify the model is ready:**

```bash
ollama list
```

You should see `qwen2.5vl:3b` listed.

---

### Step 3: Check your settings

Open `surveillant/config/settings.py` and confirm these two lines look like this:

```python
DESCRIPTION_BACKEND = "qwen-vl"      # ← this must say "qwen-vl"
OLLAMA_VLM_MODEL    = "qwen2.5vl:3b"  # ← the describer model
MARLIN_HOST         = ""             # ← this must be empty
```

This is the **default** — you probably don't need to change anything.

---

### Step 4: Run the system — descriptions happen automatically

Navigate to the `surveillant/` folder and run Phase 2 as usual:

```bash
cd "D:\college\SURVEILLANT system\surveillant"

python main.py --phase 2 --set set_1
```

Or with explicit video files:

```bash
python main.py --phase 2 --videos data/videos/video1_1.avi data/videos/video1_2.avi data/videos/video1_3.avi data/videos/video1_4.avi data/videos/video1_5.avi
```

As the system detects and identifies people, you will see lines like this in the console:

```
[SURVEILLANT] Description worker started (backend=qwen-vl, model=qwen2.5vl:3b).
[DESCRIBE] person 3a7f91c2 <- A man in a red t-shirt and black jeans (model=qwen2.5vl:3b, 8.3s)
[DESCRIBE] person b22d04e1 <- A woman in a blue jacket carrying a handbag (model=qwen2.5vl:3b, 11.1s)
```

Each `[DESCRIBE]` line means one person has been described and saved to the database. The number at the end (e.g. `8.3s`) is how long the VLM call took.

> **Note:** Descriptions are generated in a background thread and do **not** slow down the tracking/detection. The camera view continues at full speed.

---

### Step 5: Describe everyone who was already tracked (backfill)

If you ran Phase 2 before and want to describe people already in the database:

```bash
python main.py --phase 4 --describe-all
```

**Expected output:**

```
[PHASE4] describe-all using backend=qwen-vl model=qwen2.5vl:3b
[PHASE4] enqueued 47 person(s) for description.
[DESCRIBE] person 3a7f91c2 <- A man in a red t-shirt and black jeans (model=qwen2.5vl:3b, 8.3s)
[DESCRIBE] person b22d04e1 <- A woman in a blue jacket carrying a handbag (model=qwen2.5vl:3b, 11.1s)
...
[PHASE4] described 47 person(s). Stats: {'described': 47, 'failed': 0}
```

This runs synchronously and exits when done.

---

### Step 6: Search by natural language

After persons have been described, you can search:

```bash
python main.py --phase 4 --search-text "a fat man with a red t-shirt and white hat"
```

**Expected output:**

```
[SEARCH] parsed filters: {'gender': 'male', 'body_build': 'heavy', 'clothing_top': 't-shirt', 'clothing_top_color': 'red', 'headwear': 'hat', 'headwear_color': 'white'}
  Found 2 match(es):

  1. person 3a7f91c2  score=1.00  last seen cam3 @ 2026-05-28T14:32:11
     A heavy man in a red t-shirt and white hat
     snapshot: D:\college\SURVEILLANT system\surveillant\data\snapshots\3a7f91c2\crop_0.jpg

  2. person 9ff10a3c  score=0.67  last seen cam1 @ 2026-05-28T14:28:04
     A man in a red t-shirt
     snapshot: D:\college\SURVEILLANT system\surveillant\data\snapshots\9ff10a3c\crop_0.jpg
```

**More search examples:**

```bash
# Search by clothing
python main.py --phase 4 --search-text "woman in a blue jacket with glasses"

# Search by accessories
python main.py --phase 4 --search-text "anyone carrying a backpack"

# Search with negation
python main.py --phase 4 --search-text "a man without glasses in a black jacket"

# Search by build
python main.py --phase 4 --search-text "a slim woman with long hair"

# Return more results (default is top 10)
python main.py --phase 4 --search-text "man in jeans" --top-k 20
```

**Combine describe-all + search in one command:**

```bash
python main.py --phase 4 --describe-all --search-text "man in red t-shirt"
```

---

## Path B — Marlin-2B (Remote GPU)

Marlin-2B is a 2B-parameter video-VLM (Apache-2.0). It requires a **GPU** to run. SURVEILLANT runs on CPU, so you host Marlin separately and the system talks to it over HTTP.

Pick the sub-option that matches your situation:

---

### Option B1 — Google Colab (Free, T4 GPU, no setup needed)

Google Colab gives you a free T4 GPU. The Marlin server runs there; your laptop calls it.

#### On Google Colab:

**Cell 1 — Install dependencies:**
```python
!pip install -q fastapi "uvicorn[standard]" "transformers>=5.7" \
                torchcodec qwen-vl-utils pillow pyngrok
```

**Cell 2 — Copy the server script** (paste your `serve.py` content or download from your repo):
```python
# If your repo is on GitHub:
!git clone https://github.com/YOUR_USERNAME/surveillant.git
# Then copy the serve script to the current folder:
!cp surveillant/surveillant/modules/llm/marlin_server/serve.py .
```

Or paste the `serve.py` content directly into a file using `%%writefile serve.py`.

**Cell 3 — Start the server (preloads model):**
```python
import subprocess, time, os

# Start server in background
proc = subprocess.Popen(
    ["python", "serve.py", "--port", "8000", "--preload"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT
)
print("Waiting for model to load (~60 seconds)...")
time.sleep(70)
print("Server should be ready.")
```

**Cell 4 — Expose it publicly with ngrok:**
```python
from pyngrok import ngrok

# Free ngrok — may ask you to sign up (it's free)
public_url = ngrok.connect(8000)
print(f"\n>>> COPY THIS URL: {public_url.public_url}\n")
```

You'll see something like:
```
>>> COPY THIS URL: https://abc123.ngrok-free.app
```

**Keep this Colab tab open** while you use SURVEILLANT. The server shuts down when you close it.

#### On your laptop:

Open `surveillant/config/settings.py` and change:

```python
DESCRIPTION_BACKEND = "marlin"
MARLIN_HOST         = "https://abc123.ngrok-free.app"   # ← paste your Colab URL
```

Also make sure Ollama is still running (the query parser still uses it):
```bash
ollama list   # should show qwen2.5vl:3b
```

Then run as normal:
```bash
python main.py --phase 2 --set set_1
# or
python main.py --phase 4 --describe-all
```

You will see:
```
[SURVEILLANT] Description worker started (backend=marlin, model=NemoStation/Marlin-2B).
[DESCRIBE] person 3a7f91c2 <- A man in a red t-shirt and dark jeans, appears to be in his 30s (model=NemoStation/Marlin-2B, 6.2s)
```

---

### Option B2 — Your own machine with a GPU (Windows/Linux)

If you have an NVIDIA GPU on a different machine (or a desktop with a GPU at home):

**On the GPU machine:**

```bash
# Install requirements
pip install fastapi "uvicorn[standard]" "transformers>=5.7" \
            torchcodec qwen-vl-utils pillow torch

# Navigate to the server script
cd "D:\college\SURVEILLANT system\surveillant\modules\llm\marlin_server"

# Start the server (--preload downloads and loads the model at startup)
python serve.py --host 0.0.0.0 --port 8000 --preload
```

**First run downloads Marlin-2B (~4 GB) from HuggingFace. This takes a few minutes.**

You should see:
```
[MARLIN] loading NemoStation/Marlin-2B on cuda ...
[MARLIN] model ready.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**On your laptop:** Find the GPU machine's local IP (e.g. `192.168.1.50`), then edit `config/settings.py`:

```python
DESCRIPTION_BACKEND = "marlin"
MARLIN_HOST         = "http://192.168.1.50:8000"   # ← your GPU machine's IP
```

---

### Option B3 — Same machine has a GPU

If your laptop/PC itself has an NVIDIA GPU and you want to run Marlin locally:

```bash
# In a second terminal (keep this running while you use SURVEILLANT):
cd "D:\college\SURVEILLANT system\surveillant\modules\llm\marlin_server"
python serve.py --host 127.0.0.1 --port 8000 --preload
```

In `config/settings.py`:

```python
DESCRIPTION_BACKEND = "marlin"
MARLIN_HOST         = "http://127.0.0.1:8000"
```

---

## Switching Between Backends

You can switch anytime by editing just **two lines** in `surveillant/config/settings.py`:

**Use local Ollama (CPU, default):**
```python
DESCRIPTION_BACKEND = "qwen-vl"
MARLIN_HOST         = ""
```

**Use Marlin via Colab/cloud:**
```python
DESCRIPTION_BACKEND = "marlin"
MARLIN_HOST         = "https://your-url-here.ngrok-free.app"
```

No code changes, no restart of the DB needed. The worker picks up the new backend on next startup.

---

## Verifying the Marlin server is alive

Before running SURVEILLANT with the Marlin backend, check the server health:

```bash
# Replace the URL with yours:
curl http://localhost:8000/health
```

Or open it in your browser. You should see:

```json
{
  "status": "ok",
  "model": "NemoStation/Marlin-2B",
  "model_loaded": true,
  "device": "cuda"
}
```

If `model_loaded` is `false`, the server is still loading. Wait 30–60 seconds and try again.

---

## Understanding the Console Output

| Log line | What it means |
|---|---|
| `[SURVEILLANT] Description worker started (backend=qwen-vl, ...)` | Worker daemon is running. Descriptions will start appearing after the first person is identified. |
| `[DESCRIBE] person abc12345 <- A man in a red t-shirt... (8.3s)` | One person was successfully described. Takes 5–15 s for qwen-vl on CPU; 3–8 s for Marlin on GPU. |
| `[DESCRIBE] sweep enqueued 5 missing descriptions` | The background sweep found persons without descriptions and queued them. Normal. |
| `[DESCRIBE] qwen-vl HTTP failure: Connection refused` | Ollama is not running. Start it: open the Ollama app or run `ollama serve`. |
| `[DESCRIBE] marlin HTTP failure: ...` | Your Marlin GPU host is unreachable. Check if it's still running (Colab runtimes time out). |
| `[SEARCH] parsed filters: {'gender': 'male', ...}` | The query parser extracted these fields from your search text. If it's missing a field you expected, try rephrasing (the LLM-based parser is more reliable than the fallback). |
| `[SEARCH] Stage 1 returned 0 candidates.` | No persons in the DB matched the structured filter. May fall through to Stage-3 similarity search. |

---

## Frequently Asked Questions

**Q: Do I need Ollama even if I use Marlin?**
Yes — but only sometimes. The query parser uses the rule-based fast-path for common searches (no model call). When a query needs the LLM, it uses the same `qwen2.5vl:3b` via Ollama. Only the describer image calls go to Marlin.

**Q: How long will `--describe-all` take?**
Roughly: `(number of persons) × (seconds per description)`. With qwen-vl on CPU: ~10 s/person, so 50 persons ≈ 8 minutes. With Marlin on a T4 GPU: ~5 s/person, so 50 persons ≈ 4 minutes.

**Q: The search returns wrong/irrelevant results.**
Try rephrasing more specifically. The system works best with concrete colour + garment terms: "blue jacket", "red t-shirt", "black jeans". Vague terms like "casual clothes" or "normal outfit" won't filter well because they're not in the schema enums.

**Q: My Colab session timed out mid-describe. What happens to partial work?**
No data is lost. Every completed description is already in the database. When you restart (new Colab + new `MARLIN_HOST`), run `--describe-all` again — it only describes persons without a description, so already-described persons are skipped.

**Q: Can I mix: run Phase 2 with qwen-vl and then switch to Marlin for `--describe-all`?**
Yes. The DB stores which backend produced each description (`backend` column in `person_descriptions`). You can compare quality between the two by inspecting that column.

**Q: What if the JSON the VLM returns is garbage or empty?**
The system logs `[DESCRIBE] qwen-vl JSON parse failed; raw='...'` and marks the job as failed. After `MAX_DESCRIPTION_ATTEMPTS = 3` failures for the same person, it gives up and leaves that person without a description. This never crashes the system. You can re-trigger by running `--describe-all` again (it will re-enqueue failed persons).

**Q: How do I completely disable the description worker (to save CPU during a test run)?**
In `config/settings.py`:
```python
ENABLE_DESCRIPTION_WORKER = False
```
The worker thread won't start. You can still run `--describe-all` manually from Phase 4 whenever you want.

**Q: How do I add more cameras to the Marlin server's performance?**
Keep one Marlin server instance per GPU. SURVEILLANT sends one describe request at a time (the worker is single-threaded), so one instance is enough for the graduation demo. For production scale, run multiple instances behind a load balancer — but that's beyond the scope of this project.

---

## Quick Reference — All Phase 4 Commands

```bash
# Navigate to the surveillant folder first:
cd "D:\college\SURVEILLANT system\surveillant"

# Describe all persons in the DB that don't have a description yet
python main.py --phase 4 --describe-all

# Search by description
python main.py --phase 4 --search-text "a fat man with a red t-shirt"

# Search with more results
python main.py --phase 4 --search-text "man in jeans" --top-k 20

# Describe-all THEN search in one command
python main.py --phase 4 --describe-all --search-text "woman with a handbag"

# Phase 2 (live tracking) with descriptions auto-populated in background
python main.py --phase 2 --set set_1

# Phase 2 with explicit video list
python main.py --phase 2 --videos data/videos/video1_1.avi data/videos/video1_2.avi data/videos/video1_3.avi data/videos/video1_4.avi data/videos/video1_5.avi
```

---

## Settings Quick Reference

All in `surveillant/config/settings.py`:

| Setting | Default | What it controls |
|---|---|---|
| `DESCRIPTION_BACKEND` | `"qwen-vl"` | `"qwen-vl"` = local Ollama (CPU). `"marlin"` = remote GPU host. |
| `OLLAMA_VLM_MODEL` | `"qwen2.5vl:3b"` | The Ollama model used to describe images. |
| `OLLAMA_QUERY_MODEL` | `"qwen2.5vl:3b"` | The Ollama model used to parse text queries (fallback only — rules run first). Same model as the describer. |
| `MARLIN_HOST` | `""` | URL of the Marlin GPU host (e.g. `https://abc.ngrok.app`). Empty = disabled. |
| `MARLIN_TIMEOUT_SEC` | `60` | Seconds to wait for a Marlin response before failing the job. |
| `ENABLE_DESCRIPTION_WORKER` | `True` | Set `False` to disable the background worker entirely. |
| `DESCRIPTION_SWEEP_INTERVAL_SEC` | `60` | How often the worker scans for un-described persons (seconds). |
| `MAX_DESCRIPTION_ATTEMPTS` | `3` | Give up on a person after this many backend failures. |
| `ENABLE_TEXT_FALLBACK_RERANK` | `True` | Use `sentence-transformers` similarity when SQL filter finds 0 matches. |

---

*Guide written for SURVEILLANT Phase 4A+4B (2026-05-28). Covers Qwen2.5-VL (local Ollama, CPU) and Marlin-2B (remote GPU) backends.*
