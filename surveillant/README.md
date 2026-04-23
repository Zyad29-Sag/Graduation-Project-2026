# SURVEILLANT

**AI-powered multi-camera surveillance intelligence system**
*(Graduation Project)*

---

## What it Does

SURVEILLANT ingests video streams from multiple cameras, tracks every person across all cameras, builds a searchable database of face/body embeddings, and allows natural-language search queries via an LLM.

---

## Dataset

**WiseNet (Kaggle)** — 5 synchronized, overlapping surveillance camera videos.
Place the `.mp4` files in `data/videos/`.

---

## Setup

```bash
# 1. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # Linux / macOS

# 2. Install Phase 1 dependencies
pip install opencv-python ultralytics deep-sort-realtime numpy
```

---

## Phases

| Phase | Description                           | Status        |
|-------|---------------------------------------|---------------|
| 1     | Camera simulator + detection + tracking + display | ✅ Built |
| 2     | Person capture + embedding + SQLite   | 🔜 Next       |
| 3     | Photo search (cosine similarity)      | 🔜 Planned    |
| 4     | LLM description + text search         | 🔜 Planned    |
| 5     | Cross-camera person linking           | 🔜 Planned    |
| 6     | FastAPI web interface                 | 🔜 Planned    |
| 7     | React dashboard                       | 🔜 Planned    |
| 8     | Docker + PostgreSQL + pgvector        | 🔜 Planned    |

---

## Running Phase 1

```bash
cd surveillant

# Run with your video files
python main.py --phase 1 --videos data/videos/cam1.mp4 data/videos/cam2.mp4 data/videos/cam3.mp4

# Run the test suite (no video files needed)
python tests/test_phase1.py
```

Press **`q`** in the display window to quit.

---

## Project Structure

```
surveillant/
├── config/settings.py        # All thresholds & paths
├── modules/
│   ├── camera/simulator.py   # Video → simulated live stream
│   ├── detection/detector.py # YOLOv8 person detection
│   ├── tracking/tracker.py   # DeepSORT per-camera tracking
│   ├── embedding/            # (Phase 2) InsightFace embeddings
│   ├── storage/              # (Phase 2) SQLite database
│   ├── search/               # (Phase 3) Cosine similarity search
│   └── llm/                  # (Phase 4) Ollama + Qwen2.5-VL
├── display/visualizer.py     # OpenCV multi-cam grid display
├── data/videos/              # ← Place .mp4 files here
├── data/snapshots/           # Auto-created in Phase 2
├── database/                 # Auto-created in Phase 2
├── tests/                    # One test file per phase
└── main.py                   # Entry point
```

---

## Tech Stack

- Python 3.10+
- YOLOv8 (`ultralytics`)
- DeepSORT (`deep-sort-realtime`)
- InsightFace *(Phase 2)*
- SQLite → PostgreSQL + pgvector *(Phase 8)*
- Ollama + Qwen2.5-VL:2b *(Phase 4)*
- OpenCV
- FastAPI + React *(Phases 6–7)*
- Docker Compose *(Phase 8)*
