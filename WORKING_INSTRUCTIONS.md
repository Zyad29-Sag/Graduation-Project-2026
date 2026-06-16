# SURVEILLANT — Working Instructions
## How Claude Should Collaborate on This Project

> This file governs all future Claude sessions on this project.
> After every effective modification, update `DECISION_LOG.md` and save/update memory.

---

## 1. Project Identity

**SURVEILLANT** is a graduation project: a multi-camera, AI-powered person re-identification and tracking system. Code is read by professors — clarity matters.

**Dataset:** WiseNet (Kaggle) — 5 synchronized surveillance camera videos.
**Hardware target:** CPU-only (no GPU assumed).

---

## 2. Session Start Protocol

At the start of every session:
1. Read `MEMORY.md` (index) and any relevant memory files.
2. Read `DECISION_LOG.md` — understand what has been implemented and why.
3. Read `SURVEILLANT_ARCHITECTURE_REPORT.md` — current system state.
4. Check `git status` and `git log --oneline -5` to see what changed since last time.
5. Confirm current phase and what is next before touching any code.

---

## 3. Phase Discipline

The project follows a build-spec (`SURVEILLANT_BUILD_SPEC.md`) with strict phase gates:
- Never implement a later phase while an earlier one is untested.
- After each phase, **stop and wait** for user confirmation before proceeding.
- The Enhancement Proposal (`ENHANCEMENT_PROPOSAL.md`) is the improvement roadmap, tracked by Part number (1–10). The current last-implemented Part is confirmed in `DECISION_LOG.md`.

**Current status as of 2026-05-28:**
- All Enhancement Proposal Parts 1–8 implemented and validated.
- Parts 9 (spatio-temporal constraints) and 10 (LLM description matching) are next.

---

## 4. After Every Effective Modification

"Effective modification" = any change to logic, a model, a threshold, an algorithm, or a new feature (not typo fixes or formatting).

**Steps required after every effective modification:**
1. Update `DECISION_LOG.md` — add an entry under the correct phase/part number with:
   - What was changed and in which file(s).
   - Why (motivation, bug being fixed, enhancement being added).
   - What was removed/replaced (old approach) if applicable.
   - Expected outcome / measured result if available.
2. Update `SURVEILLANT_ARCHITECTURE_REPORT.md` if the architecture changed.
3. Update memory files if user preferences or project state changed.
4. Commit with a clear message (when user asks).

---

## 5. Code Standards

- All functions and classes must have docstrings (graduation project — professors read this).
- Type hints everywhere.
- No magic numbers — all thresholds and sizes must be in `config/settings.py`.
- Error handling: never let one camera's error crash others.
- Performance: never call LLM or embedder on every frame.
- Comments only when the WHY is non-obvious.

---

## 6. File Ownership Map

| File | Purpose |
|---|---|
| `config/settings.py` | All thresholds, paths, model names |
| `modules/camera/simulator.py` | CameraSimulator — reads video files |
| `modules/detection/detector.py` | PersonDetector — YOLOv8-seg |
| `modules/tracking/tracker.py` | PersonTracker — ByteTrack |
| `modules/preprocessing/quality_gate.py` | CropQualityGate — blur/dark/size filter |
| `modules/preprocessing/enhancement.py` | FrameEnhancer — CLAHE + auto-gamma |
| `modules/preprocessing/masking.py` | Mask application + track-mask association |
| `modules/embedding/embedder.py` | PersonEmbedder — OSNet x1.0 + Market-1501 |
| `modules/embedding/gallery.py` | GalleryManager — pose-aware gallery decisions |
| `modules/storage/database.py` | Database — SQLite WAL + callback hooks |
| `modules/search/searcher.py` | PersonSearcher — FAISS + SQLite fallback |
| `modules/search/faiss_index.py` | FAISSIndex — in-memory vector index |
| `modules/reconciliation/worker.py` | ReconciliationWorker — background merge daemon |
| `modules/llm/describer.py` | PersonDescriber — Ollama + Qwen2.5-VL (Phase 4) |
| `modules/face/face_analyzer.py` | FaceAnalyzer — InsightFace + watchlist + "returning" (Part 11) |
| `modules/face/ethnicity_classifier.py` | EthnicityClassifier — ResNet18 (Part 11, optional) |
| `modules/face/face_searcher.py` | FaceSearcher — face-image search over the isolated store (Part 11) |
| `modules/detection/glasses_detector.py` | GlassesDetector — ResNet18 (Part 11, optional) |
| `modules/violence/violence_detector.py` | ViolenceDetector — CNN-LSTM (Part 11, optional) |
| `modules/violence/alerts.py` | Violence alert log/clip/email helpers (Part 11) |
| `main.py` | Entry point — Phase 1 and Phase 2 runners |
| `display/visualizer.py` | GridDisplay + ColorRegistry |

---

## 7. Key Invariants (Never Break These)

- `DETECTION_CONF = 0.10` — pass all detections to ByteTrack; do NOT raise this.
- High-confidence gating for embedding crops happens via IoU ≥ 0.7 in `main.py`, NOT via conf threshold.
- Quality gate (`CropQualityGate`) runs on gallery-update path only. NOT on identification path.
- View-coverage gate (`MIN_VIEW_COVERAGE_FOR_MATCHING`) is for **reconciliation only** — never in the searcher.
- SQLite is always the source of truth. FAISS is a redundant cache only.
- Database callback hooks (`on_embedding_added`, `on_merge`) fire AFTER the SQLite transaction commits.
- `Database.propose_merge()` uses upsert — never creates duplicate proposals.
- `FORCE_ACCEPT_MAX_DISTANCE` must never exceed `1 − BODY_MATCH_THRESHOLD`. Force-accept bypasses the diversity gate — if set looser than the identification threshold, wrong-person crops can pollute a gallery even after the identification logic correctly rejects them.
- **(Part 11) Face embeddings live ONLY in the `face_embeddings` table — never in `person_embeddings`.** InsightFace vectors are 512-d (same as OSNet body) and the body searcher / FAISS / reconciliation pool by *dimension*, not by `embedding_type`; mixing them corrupts body identity. `add_face_embedding` must never fire the `on_embedding_added` FAISS hook.
- **(Part 11) Face & violence are ADDITIVE** — they never bind/merge/re-score a `person_id`. Cross-camera identity stays 100% body (OSNet/ByteTrack); face only contributes attributes, watchlist names, a "returning" badge, and the `--face-photo` search.
- **(Part 11) Face & violence default OFF** (`ENABLE_FACE_ANALYSIS`, `ENABLE_VIOLENCE_DETECTION`) and degrade gracefully when a model file is missing. With flags off, runtime behavior is identical to pre-Part-11.
- **(Part 11) The violence worker is fully isolated** — it writes only to `violence_log.json` / alert clips / email and never touches person tables, embeddings, registry, or tracking dicts; it must never block detection/embedding.
- **(Part 11) No secrets in code** — violence email credentials come from environment variables (`SURVEILLANT_ALERT_*`) only; the team's old copy leaked a live Gmail app password (revoke it).

---

## 8. What Is NOT In Scope

- IR/infrared cameras (mentioned in Enhancement Proposal §3.3 — explicitly out of scope for this project).
- GPU optimization (CPU-only deployment target).
- Docker / React frontend — those are later phases, don't implement early.

---

## 9. Collaboration Style

- If the user says "implement X", read relevant files, confirm the approach in one sentence, then code.
- If something is ambiguous or has architectural trade-offs, state the trade-off and ask before coding.
- Keep responses short and concrete. No padding.
- After writing code, state what was changed and what the user should test.

---

*Last updated: 2026-06-16 — Part 11 (face & violence integration) added.*
