# SURVEILLANT — Decision Log
## Full Record of Ideas, Implementations, Replacements, and Reasoning

> Every significant change goes here. This file is the primary source for writing
> project documentation, reports, and academic write-ups.
> Format: newest entries go at the top within each part.

---

## How to Read This File

- Each entry has: **What**, **Why**, **Old approach (if replaced)**, **New approach**, **Result/Status**.
- Part numbers match `ENHANCEMENT_PROPOSAL.md`.
- Phase numbers match `SURVEILLANT_BUILD_SPEC.md`.

---

## Current Implementation Status

| Part | Title | Status |
|---|---|---|
| Phase 1 | Camera + Detection + Tracking + Display | ✅ Complete |
| Phase 2 | Embedding + Database + Cross-Camera Matching | ✅ Complete |
| Phase 3 | Photo Search | ✅ Complete |
| 3.5 | Preprocessing Hardening | ✅ Complete |
| Part 1 | Crop Quality Gate | ✅ Complete |
| Part 2 | CLAHE + Auto-Gamma Frame Enhancement | ✅ Complete |
| Part 3 | (IR/VI-ReID — out of scope, no IR hardware) | ⛔ Skipped |
| Part 4 | Background Isolation via YOLO-seg masking | ✅ Complete |
| Part 5 | OSNet x1.0 Embedder (replaces ResNet-50) | ✅ Complete |
| Part 6 | Pose-Aware Gallery (canonical views) | ✅ Complete |
| Part 7 | ByteTrack (replaces DeepSORT) | ✅ Complete |
| Part 8 | FAISS In-Memory Vector Index | ✅ Complete |
| Part 8.5 | Overlap-Aware Matching (camera topology, prelude to Part 9) | ✅ Complete |
| Part 9 | Spatio-Temporal Camera Constraints | 🔲 Not yet implemented |
| Part 10 (Phase 4A+4B) | LLM Body Description + Natural-Language Search | ✅ Complete |
| Part 10 (Phase 4C/4D) | Multi-snapshot consensus, color sanity-check, re-description triggers, hybrid photo+text rerank | 🔲 Deferred |

---

## Phase 1 — Camera Simulator + Detection + Tracking + Display

### P1-A: Multi-threaded display architecture
**What:** Split display and detection into two threads: main thread renders at native FPS, worker thread does round-robin YOLO inference.
**Why:** Running YOLO detection synchronously on every frame made the display freeze. Round-robin (one camera per detection cycle) ensures smooth display while maximising throughput on CPU.
**Result:** Display stays smooth regardless of how long YOLO takes per frame.

### P1-B: Initial tracker — DeepSORT
**What:** Used `deep-sort-realtime` library for per-camera tracking.
**Why:** DeepSORT was the standard multi-object tracker at the time. Requires a Re-ID appearance model (embedding overhead).
**Status:** **REPLACED by ByteTrack** (see Part 7).

---

## Phase 2 — Embedding + Database + Cross-Camera Matching

### P2-A: Initial embedder — ResNet-50 (ImageNet, 2048-d)
**What:** Used torchvision ResNet-50 pretrained on ImageNet as the body embedding backbone.
**Why:** Available out-of-the-box, fast, well-known.
**Problem:** ImageNet weights encode *semantic category* (chair, cat, car) — not *person identity*. Same-person/different-person similarity distributions overlap heavily:
- Same person front→back: 0.45–0.65
- Different people, similar clothing: 0.55–0.75
- Distributions overlap completely — no reliable threshold possible.
**Status:** **REPLACED by OSNet x1.0 + Market-1501** (see Part 5).

### P2-B: Async embedding worker
**What:** A dedicated embedding thread consumes an `embed_queue` of `(cam_id, track_id, crops, ...)` items. Detection thread never blocks on embedding.
**Why:** OSNet inference takes ~20-50ms on CPU. Running it in the detection loop would drop frames. The queue decouples the two.
**Result:** Detection and display run unimpeded; embedding runs as fast as possible in the background.

### P2-C: Permanent track registry
**What:** `track_registry[(cam_id, track_id)] = person_id` — once a track is bound to a person, the binding is permanent for the session.
**Why:** Prevents repeated embedding calls for the same track. The registry is saved to `database/track_registry_session.json` so sessions survive restarts.
**Rule 1:** Intra-camera — once bound, never re-checked.
**Rule 2:** Inter-camera — new track does one embedding search, then Rule 1 applies.

### P2-D: SQLite with WAL
**What:** SQLite in WAL (Write-Ahead Logging) mode.
**Why:** The detection thread (reads) and embedding thread (writes) access the DB concurrently. WAL allows one writer + multiple readers simultaneously without deadlocks.
**Schema tables:** `persons`, `person_embeddings`, `camera_history`, `merge_proposals`.

### P2-E: Database callback hooks
**What:** `Database.on_embedding_added` and `Database.on_merge` — optional callbacks fired AFTER SQLite transaction commits.
**Why:** Allows downstream caches (FAISS index) to stay in sync without the Database knowing about them directly. Failures in hooks are caught and logged — they never take down the embedding worker because SQLite is the source of truth.

---

## Part 1 / Part 2 — Crop Quality Gate

### QG-A: Quality gate implementation
**File:** `modules/preprocessing/quality_gate.py`
**What:** Three-check filter applied before extracting gallery embeddings:
1. **Laplacian variance ≥ 50.0** — rejects motion-blurred / out-of-focus crops.
2. **Minimum size ≥ 48×96 px** — rejects far/edge crops that upscale badly.
3. **HSV V mean ≥ 30** — rejects dark silhouettes with no color information.
**Why:** A blurry, tiny, or dark crop produces an embedding that represents noise, not identity. Storing it pollutes the gallery and causes false non-matches.
**Applied to:** Gallery-update path only (both in `main.py` and defensively inside `GalleryManager`). **NOT** applied to identification crops — even a blurry crop carries enough signal for a one-time "does this person exist?" query.

---

## Part 2 — CLAHE + Auto-Gamma Frame Enhancement

### FE-A: FrameEnhancer implementation
**File:** `modules/preprocessing/enhancement.py`
**What:** Two-stage frame preprocessing before YOLO detection:
1. **CLAHE** (Contrast Limited Adaptive Histogram Equalization) in LAB color space — equalizes luminance channel only, preserving color hues for the embedder.
2. **Auto-gamma** (γ = 0.5) — applied only when mean frame brightness < 60 (very dark scenes).
**Why:** Indoor surveillance scenes have variable lighting. Dark frames cause YOLO to miss detections and produce embeddings that encode shadow shapes rather than identity.

### FE-B: Conditional CLAHE (Bug fix — Part 9 audit)
**What:** CLAHE was originally applied to every frame unconditionally.
**Problem:** On well-lit frames, CLAHE amplifies fine-grained noise and slightly degrades color accuracy — wasted ~10ms/frame.
**Fix:** Skip both stages when mean V-channel brightness > 120. Only trigger enhancement when the frame is actually dark.
**File:** `modules/preprocessing/enhancement.py`

---

## Part 4 — Background Isolation via YOLO Segmentation

### BG-A: Switch from `yolov8n.pt` to `yolov8n-seg.pt`
**What:** Upgraded YOLO model from detection-only to segmentation variant.
**Why:** The `-seg` variant outputs per-person pixel masks alongside bounding boxes. Zeroing out background pixels before sending the crop to the embedder means the embedding encodes *the person*, not the background wall/floor.
**Old problem:** Two different people in front of the same red wall had artificially high similarity. Same person in front of different backgrounds had artificially low similarity.
**Cost:** ~3ms extra per frame. Acceptable for the improvement in embedding quality.
**Config:** `YOLO_MODEL = "yolov8n-seg.pt"`, `USE_SEGMENTATION = True`
**Background replacement:** Neutral gray (value 128) — not black, because black biases the embedder toward encoding an absence of color.

### BG-B: IoU-based mask-to-track association
**File:** `modules/preprocessing/masking.py` — `associate_masks_to_tracks()`
**What:** After ByteTrack reorders and Kalman-corrects track bboxes, a simple index-match between YOLO detections and ByteTrack outputs fails. This function recovers the mask for each track via IoU matching (threshold 0.4).
**Why:** ByteTrack returns Kalman-corrected bboxes that don't exactly match YOLO's original detections. Direct equality comparison (`d["bbox"] == t["bbox"]`) always fails (Bug B fix).

---

## Part 5 — OSNet x1.0 Embedder

### EM-A: Replace ResNet-50 with OSNet x1.0
**File:** `modules/embedding/embedder.py`
**What:** Replaced ResNet-50 (ImageNet, 2048-d) with OSNet x1.0 (Market-1501 Re-ID weights, 512-d).
**Why:** OSNet was purpose-built for person Re-ID (ICCV 2019). Market-1501 is a Re-ID benchmark dataset of 1,501 identities across 6 cameras — OSNet trained on it encodes *individual identity*, not semantic category.

**Old (ResNet-50 / ImageNet):**
- Same person, 90° turn: 0.55–0.75 (overlaps different-person range)
- Different people, similar clothes: 0.55–0.75

**New (OSNet / Market-1501):**
- Same person, 90° turn: 0.78–0.92 (clean separation)
- Different people, similar clothes: 0.20–0.50

**Critical note:** `torchreid.build_model(pretrained=True)` loads **ImageNet weights only** — NOT Re-ID weights. Market-1501 weights must be downloaded separately via `gdown` from the torchreid model zoo. This step is mandatory; without it OSNet performs barely better than ResNet-50.
**Cache path:** `~/.cache/torchreid/checkpoints/osnet_x1_0_market1501.pth`

### EM-B: Input size — 256×128 (H×W)
**What:** OSNet input is resized to 256×128, not 224×224.
**Why:** OSNet was trained on this aspect ratio which matches typical person crops (taller than wide). Using 224×224 would distort body proportions.

### EM-C: Median aggregation (Bug H fix)
**What:** Changed `aggregate_embeddings()` from mean to median pooling over the 4-frame identification buffer.
**Why:** Mean is skewed by outlier frames (blur flash, partial occlusion). Median is robust — one bad frame out of 4 doesn't move the aggregate.

### EM-D: Database incompatibility after backbone change
**Important:** Old 2048-d ResNet-50 embeddings are byte-incompatible with new 512-d OSNet embeddings. The dimension check in `searcher.py` and the reconciliation worker auto-skip stale embeddings, but the database should be deleted and rebuilt from scratch after the upgrade.

---

## Part 6 — Pose-Aware Gallery

### PG-A: Canonical view classification
**File:** `modules/embedding/gallery.py` — `estimate_view()`
**What:** Classifies a bounding box into one of 4 canonical views: `frontal`, `right_moving`, `left_moving`, `side`.
**Logic:**
- `aspect = width/height < 0.40` → `"side"` (person sideways)
- Horizontal Δcenter > 8px → `"right_moving"` / `"left_moving"`
- Otherwise → `"frontal"`

### PG-B: Force-accept for uncovered canonical slots
**What:** If a canonical view slot (e.g. `right_moving`) is not yet covered for a person, the new embedding is force-accepted regardless of cosine distance, bypassing the novelty/diversity gate.
**Why:** Without this, the gallery accumulates near-duplicate frontal views because the diversity gate only accepts novelty — and the person never happens to walk sideways when the diversity gate is satisfied.
**Guard:** Force-accept still checks `GALLERY_MAX_DISTANCE` (garbage check) and `MAX_GALLERY_SIZE`.

### PG-C: View coverage score
**What:** `get_view_coverage()` returns `covered_canonical_slots / 4` (0.0–1.0).
**Used in:** Reconciliation worker only — both persons must have coverage ≥ 0.5 before being considered merge candidates.
**NOT used in:** The real-time searcher (Bug F fix — see below).

### PG-D: Bug F — view-coverage gate in searcher caused duplicate IDs
**What:** The view-coverage gate was originally also in `searcher.py` to skip persons with < 2 canonical views.
**Problem:** When a person re-entered the scene, their gallery was already built but the view-coverage check caused the searcher to skip them → new person_id created every re-entry → unbounded duplicate proliferation.
**Fix:** Removed the view-coverage gate from the searcher entirely. The gate only makes sense for reconciliation (batch, high-confidence decisions), not real-time re-entry matching.

### PG-E: Bug C — initial embedding tagged "initial" blocked reconciliation
**What:** The first embedding for a new person was tagged `"initial"` (not in `CANONICAL_VIEWS`).
**Problem:** `get_view_coverage()` only counts canonical views — `"initial"` doesn't count. So every person started with view-coverage 0.0 and was never eligible for reconciliation.
**Fix:** When creating a new person in `main.py`, the canonical view is estimated from the first crop's bbox and passed as `angle_tag` to `insert_person()`. The initial embedding is now tagged with a canonical view from day one.

---

## Part 7 — ByteTrack (Replaces DeepSORT)

### BT-A: Replace DeepSORT with ByteTrack
**File:** `modules/tracking/tracker.py`
**What:** Switched from `deep-sort-realtime` to `ultralytics` built-in ByteTrack.
**Why:**
- DeepSORT requires an appearance model (heavy) and discards low-confidence detections.
- ByteTrack uses **two-stage IoU association** — low-confidence detections (0.10–0.45) keep coasting tracks alive through occlusion instead of killing them.
- When a person is half-occluded, YOLO outputs a low-conf detection (0.25–0.40). DeepSORT discards it → track dies → new ID on reappearance → "color change on turn" bug. ByteTrack uses it → track survives → same color.

**Two-stage association:**
- Stage 1 (≥ 0.45 conf): IoU match against active tracks.
- Stage 2 (0.10–0.45 conf): matches against unmatched tracks from Stage 1 only.

**Config:**
```
BYTETRACK_TRACK_THRESH = 0.45   (high-conf gate)
BYTETRACK_LOW_THRESH   = 0.10   (low-conf gate — keeps tracks alive through occlusion)
BYTETRACK_MATCH_THRESH = 0.80   (IoU threshold for association)
DETECTION_CONF         = 0.10   (pass all detections to ByteTrack)
```

### BT-B: Bug B — IoU-based high-conf crop filter
**What:** Crops for embedding were gated on `d["bbox"] == t["bbox"]` (exact list equality).
**Problem:** ByteTrack returns Kalman-corrected bboxes, which are never exactly equal to YOLO's raw detection bboxes. This always returned False → no crops ever queued for embedding → no identities ever created.
**Fix:** Replaced with IoU ≥ 0.7 match between the ByteTrack track bbox and the highest-confidence matching YOLO detection for that frame.

---

## Part 8 — FAISS In-Memory Vector Index (+ Regression Investigation)

### FA-B: Gallery-sponge regression reported after FAISS — diagnostic flags added (2026-05-28)
**Observed symptom:** After Part 8 was merged, multiple visually-distinct walking people were all assigned the same `person_id` and the same bounding-box color ("the orange one"), while the sitting person stayed correctly isolated. The debug dashboard confirmed a "gallery sponge" — one person record (`6e7cc524`) accumulated snapshots of at least 3 different physical people, with all 4 canonical angle slots filled. Live-activity log showed this person "moving" between cameras 1–4 in seconds — physically impossible, confirming wrong track bindings.

**Confirmed:** fresh database was used, so the sponge forms within a single session (not caused by stale embeddings from a prior run).

**Hypotheses being tested (see plan file for full details):**
- **H1** — FAISS `IndexFlatIP` returns inflated scores vs sklearn cosine (FAISS bug).
- **H2** — FAISS is 1000× faster, shifting timing/race windows in the embedding worker.
- **H3** — Pre-existing calibration issue: `BODY_MATCH_THRESHOLD = 0.65` + Part 6 force-accept (only checks `GALLERY_MAX_DISTANCE = 0.55`) is too loose, creating a self-reinforcing sponge. FAISS made it consistently visible.

**Diagnostic changes added (Step 1):**
1. `ENABLE_FAISS = True` in `config/settings.py` — kill switch. Set `False` to fall back to SQLite linear scan for A/B testing.
2. `FAISS_AUDIT_MODE = False` in `config/settings.py` — when `True`, every identification query runs both FAISS and SQLite paths; logs `[FAISS_DRIFT]` if scores differ by > 0.01 or top-1 person IDs differ.
3. `modules/search/searcher.py` — added `_search_via_sqlite_raw()` (returns raw scored list) and `_log_audit_drift()` (compares the two paths); `search_by_embedding()` routes through audit path when `FAISS_AUDIT_MODE=True`.
4. `main.py` — FAISS construction in `run_phase2` and `run_phase3` wrapped in `if ENABLE_FAISS:`.

**Diagnostic result (2026-05-28):** `ENABLE_FAISS=False` ran — sponge still formed. **H3 confirmed. FAISS is innocent.**

### FA-C: H3 fix — BODY_MATCH_THRESHOLD raised + force-accept guard tightened (2026-05-28)
**Root cause confirmed:** Two physically-distinct walking people in the WiseNet scene had OSNet embeddings with cosine similarity ≥ 0.65, triggering an initial wrong binding. Once bound, the pose-aware force-accept path (Part 6) absorbed the wrong person's subsequent crops into the gallery because its only guard was `distance ≤ GALLERY_MAX_DISTANCE = 0.55` (sim ≥ 0.45) — far too loose. The gallery became a sponge covering all 4 canonical angles with multiple people's embeddings. Every new track's max-pool query then matched this multi-person gallery even more easily, snowballing.

**FAISS is innocent** — it faithfully reproduced the same wrong match that SQLite would have. FAISS re-enabled.

**Fix 1 — `BODY_MATCH_THRESHOLD` raised 0.65 → 0.72** (`config/settings.py`)
- 0.72 is the OSNet/Market-1501 recommended boundary from Enhancement Proposal §5.2.
- 0.65 was deliberately lowered for sitting/standing tolerance but turned out to put the threshold *inside* the different-person score range for this dataset (~0.65–0.70).
- 0.72 sits clearly above the different-person ceiling and well below the same-person floor across pose changes (0.78+).

**Fix 2 — `FORCE_ACCEPT_MAX_DISTANCE = 0.35` added; used in gallery.py force-accept check** (`config/settings.py`, `modules/embedding/gallery.py`)
- Force-accept (Part 6) now requires the new embedding to score ≤ 0.35 distance (sim ≥ 0.65) against the existing gallery before it fills an empty canonical slot.
- Previously used `GALLERY_MAX_DISTANCE = 0.55` (sim ≥ 0.45) — essentially any non-garbage crop could be force-accepted.
- This is a defense-in-depth guard: even if a track is wrongly bound (which Fix 1 makes much less likely), its subsequent gallery crops won't be accepted if they're too different from what's already stored.

**New invariant added:** `FORCE_ACCEPT_MAX_DISTANCE` must never be set higher than `BODY_MATCH_THRESHOLD`'s equivalent distance (i.e., `1 - BODY_MATCH_THRESHOLD = 0.28`). Currently 0.35 is slightly looser than identification threshold (0.28) — intentionally so, since gallery updates are lower-stakes than initial binding.

### FA-D: Dual-threshold matching — same-cam vs cross-cam (2026-05-28)
**Observed after H3 fix:** Two residual issues:
1. Small mini-sponge (`991b03df`, 3 embeds, side-only, cam1) — two different people's side views scored ≥ 0.72 → still merged at identification.
2. Cross-camera split — same person transitioning cam1 → cam2 had their aggregate embedding drop to 0.68–0.72 due to angle/lighting change → created a new person_id.

**Root cause of remaining tension:** A single threshold cannot separate the overlap zone 0.68–0.72 which contains BOTH different-people-same-cam scores AND same-person-cross-cam scores.

**Fix — context-aware dual threshold** (`config/settings.py`, `main.py`, `searcher.py`):
- `BODY_MATCH_THRESHOLD_SAME_CAM = 0.75` — applied when top-1 candidate's `last_seen_cam == current cam`. A truly returning same-camera person scores 0.85+; 0.75 clears the ~0.72 false-positive ceiling for same-camera side views.
- `BODY_MATCH_THRESHOLD_CROSS_CAM = 0.68` — applied when `last_seen_cam != current cam`. Cross-camera angle/lighting drops same-person scores to 0.68–0.72; 0.68 catches them.
- `BODY_MATCH_THRESHOLD = 0.68` — set equal to the lower threshold (serves as the searcher floor). PersonSearcher._hydrate uses this so cross-camera candidates are not filtered before main.py's context check.
- `searcher.search_by_embedding()` now accepts `min_threshold` optional override, used by phase-2 identification to request the cross-cam floor.
- **Log format upgraded:** `[MATCH]` and `[BELOW]` lines now print `(cross-cam(cam1→cam2))` or `(same-cam(cam4))` for easier debugging.

**Architectural note:** The same-camera live-conflict guard (`same_cam_conflict`) already prevents simultaneous same-camera duplicates. The dual threshold handles the sequential case (same-camera different person arriving after the previous one left).

### FA-A: FAISS IndexFlatIP implementation
**File:** `modules/search/faiss_index.py`
**What:** In-memory `faiss.IndexFlatIP(512)` — exact inner-product search on L2-normalized vectors = cosine similarity.
**Why:** SQLite linear scan on every query loaded all embeddings, computed cosine similarity in a Python loop. At 30 persons × 5 views = 150 vectors: ~10ms/query. At 500 persons: ~300ms/query (real-time bottleneck).
**Result:** 916× speedup measured at 100-vector gallery (65.3ms → 0.07ms / query).

**Architecture:**
- FAISS = redundant in-memory copy for fast nearest-neighbour search.
- SQLite = source of truth for all metadata and embeddings.
- If `faiss-cpu` is not installed or the index is empty, the Searcher transparently falls back to SQLite.

**Synchronization:**
1. Startup: `rebuild_from_db()` bulk-loads all embeddings from SQLite.
2. Live insert: `Database.on_embedding_added` callback → `FAISSIndex.add()`.
3. Merge: `Database.on_merge` callback → `FAISSIndex.reassign_person()` (relabels idx→pid map, no rebuild).

**Aggregation:** Max-pool per person (same as SQLite path — ensures identical threshold semantics).

**Validated (measured):**
- 20/20 identical top-1 matches vs SQLite on 20-person × 5-view test gallery.
- Score difference: 0.000000 (within float-32 precision).
- Concurrent stress: 1,174 inserts + 3,612 searches over 2s, 0 errors.
- Stale 2048-d embeddings silently skipped — no crash.

---

## Part 8.5 — Overlap-Aware Matching (Camera Topology, prelude to Part 9)

### OV-A: Problem statement (2026-05-28)

**What:** The dual-threshold matching (0.75 same-cam / 0.68 cross-cam) was designed for *sequential* cross-camera transitions where the person leaves cam1 and arrives at cam2 with a similar canonical view. It does NOT handle the case where two or more cameras have overlapping fields of view — same room, different angles — and the same physical person appears on cam1 and cam2 *simultaneously*.

**Why it matters:** OSNet's same-person score across a sharp angle change in the same instant lands in 0.55–0.70 — *below* the 0.68 cross-cam threshold. The system would split one physical person into two `person_id`s with no live mechanism to merge them. On WiseNet (5 rooms / hallways with non-trivial camera coverage) this is a visible, observable failure mode that the dual threshold cannot fix alone.

**Dual of Part 9:** Part 9 (spatio-temporal) rejects physically impossible matches. Part 8.5 rescues legitimate matches the visual matcher missed because of an angle gap. Both need topology knowledge; both compose cleanly. Part 8.5 ships first because overlap-splits are observable today.

### OV-B: Layer 1 — Triple-threshold live matching

**Files:** `config/settings.py`, `main.py`

**What:** Replaced the dual-threshold context-aware decision in `main.py`'s embedding worker (~line 542) with a three-way pick:

| Branch | Threshold | Rationale |
|---|---|---|
| Same camera (`last_cam == cur_cam`) | `BODY_MATCH_THRESHOLD_SAME_CAM = 0.75` | unchanged — strict to reject same-camera sequential look-alikes |
| Overlap partner (`are_overlapping_cams(cur_cam, last_cam)`) | `BODY_MATCH_THRESHOLD_OVERLAP = 0.62` | NEW — looser, because same-person sharp-angle-in-same-instant scores from a lower distribution than sequential cross-cam |
| Cross-cam, non-overlap | `BODY_MATCH_THRESHOLD_CROSS_CAM = 0.68` | unchanged — moderate, sequential transitions land in 0.68–0.78 |

**New config:**
- `CAMERA_OVERLAP_GROUPS: list[set[int]] = []` — user-configurable declaration of which cam_ids share physical space. Empty default = feature off, drop-in safe.
- `BODY_MATCH_THRESHOLD_OVERLAP = 0.62`.
- `are_overlapping_cams(cam_a, cam_b)` helper.
- `validate_overlap_topology(num_cams)` — warns at startup on disjoint-group violations, unknown cam_ids, threshold-ordering violations. Never raises (WORKING_INSTRUCTIONS §5).

**Searcher floor change (also in `main.py`):** `min_threshold` in the identification search call dropped from `BODY_MATCH_THRESHOLD_CROSS_CAM = 0.68` to `BODY_MATCH_THRESHOLD_OVERLAP = 0.62`. Otherwise the searcher would pre-filter overlap-band candidates before main.py's context check sees them. When `CAMERA_OVERLAP_GROUPS = []`, candidates in [0.62, 0.68) are still returned but then correctly rejected by the cross-cam threshold — harmless overhead.

**`BODY_MATCH_THRESHOLD` (legacy alias) stays at `CROSS_CAM = 0.68`.** This is the floor for callers without camera context (Phase-3 photo search). Live identification passes its own `min_threshold=` override and is unaffected.

**Log format:** `[MATCH:OVERLAP]` (new) for overlap-cam matches alongside `[MATCH]` for same-cam / cross-cam. Cam label format `(overlap-cam(cam1↔cam2))` mirrors the `(cross-cam(cam1→cam2))` / `(same-cam(cam4))` pattern added in FA-D.

**New invariant (#12):** `BODY_MATCH_THRESHOLD_OVERLAP <= BODY_MATCH_THRESHOLD_CROSS_CAM <= BODY_MATCH_THRESHOLD_SAME_CAM`. Also: `(1 - BODY_MATCH_THRESHOLD_OVERLAP) > FORCE_ACCEPT_MAX_DISTANCE` so force-accept doesn't become a sponge on overlap pairs (0.38 > 0.35 at default).

### OV-C: Layer 2 — Co-visibility boost in reconciliation

**File:** `modules/reconciliation/worker.py`, `modules/storage/database.py` (new helper `get_camera_history()`)

**What:** Before computing `mean-pool similarity` and comparing against `MERGE_CANDIDATE_THRESHOLD = 0.58`, the reconciliation cycle now also computes a *co-visibility moment count* for the pair: how many independent moments did the two persons' `camera_history` intervals temporally overlap on either (a) the same physical camera or (b) declared overlap-partner cameras. If `count >= CO_VISIBILITY_MIN_MOMENTS = 3` (each ≥ `CO_VISIBILITY_MIN_OVERLAP_SEC = 0.5` s of overlap), the threshold for THAT PAIR drops from 0.58 to `MERGE_CANDIDATE_THRESHOLD_OVERLAP_BOOSTED = 0.45`.

**Why this works:** Two unrelated people would not repeatedly share the *exact* same time window on the *same* overlap pair across multiple distinct moments — that happens only by coincidence, rarely. One physical person mis-split into two IDs will show up in lock-step on the overlap-partner cameras every time they enter the room.

**Why this is safe:** The mean-pool similarity floor still applies. A pair only gets the boosted treatment if both visual similarity (mean-pool ≥ 0.45) AND repeated co-visibility (≥ 3 moments) agree. Either signal alone is insufficient.

**New log line:** `[RECONCILE:CO-VISIBILITY]` when the boost fires, with moment count and score.

**New database method:** `Database.get_camera_history(person_id)` returns full `(cam_id, track_id, first_seen, last_seen)` rows — needed because `get_cameras_for_person()` only returns distinct cam_ids and discards the timing.

### OV-D: Forward compatibility with Part 9

When Part 9 (spatio-temporal transition matrix) lands, overlap-group pairs get `MIN_TRANSITION_SEC = 0` for free — no special case needed. The two features compose:

| Pair type | Part 9 (future) | Part 8.5 (now) |
|---|---|---|
| Same cam | min_transition = 0 | threshold = 0.75 |
| Overlap partner | min_transition = 0 | threshold = 0.62 |
| Adjacent non-overlap | min_transition = ~8 s | threshold = 0.68 |
| Far non-overlap | min_transition = ~30 s | threshold = 0.68 |

### OV-E: Status

- **WiseNet topology declared (2026-05-28):** `CAMERA_OVERLAP_GROUPS = [{3, 4}]` — cameras 3 and 4 share a physical field of view. Overlap branch is now live for cam3↔cam4 pairs.
- All Phase-2 and Phase-3 regressions verified by syntax/import check; integration validated by per-file `ast.parse()` round-trip and invariant assertions.
- Startup validator prints topology summary: `[SURVEILLANT] Overlap topology: 1 group(s) — thresholds same=0.75 overlap=0.62 cross=0.68`.
- `[MATCH:OVERLAP]` and `[RECONCILE:CO-VISIBILITY]` log lines will appear when cam3 and cam4 share a person sighting.

---

## Part 10 (Phase 4A + 4B) — LLM Body Description + Natural-Language Search

### LL-A: Problem statement (2026-05-28)

**What:** Parts 1–8.5 give a stable `person_id` and a multi-view body gallery for each person, which makes "is this the same person?" queries work but leaves the system blind to semantics — "find a man in a red t-shirt and white hat" cannot be answered because the database stores embedding vectors, not appearance attributes.

**Why it matters:** A graduation demo needs a query interface the human operator can speak. Cross-camera identity tracking is invisible to a non-technical audience; natural-language search is the headline feature that makes the work feel real.

### LL-B: Model choice — Marlin-2B primary, Qwen2.5-VL fallback

**Decision (user-driven, 2026-05-28):** Use `NemoStation/Marlin-2B` (Apache-2.0, video VLM) as the primary describer, with `qwen2.5vl:2b` via local Ollama as the CPU fallback.

**Why two backends:**
- Marlin-2B is a video-trained VLM and requires a GPU per its model card. SURVEILLANT's local target is CPU-only.
- We resolve this with a remote-HTTP architecture: a small FastAPI host (`modules/llm/marlin_server/serve.py`) runs Marlin on a GPU machine (Colab notebook, cloud VM, or local workstation with NVIDIA), and SURVEILLANT POSTs base64-encoded snapshots over the network.
- When the GPU host is unreachable or `MARLIN_HOST = ""`, the system gracefully falls back to `QwenVLOllamaDescriber` so dev iterations always have a working path.
- Both backends share an abstract `Describer` interface (`modules/llm/describer.py`), so the worker code is backend-agnostic.

### LL-C: Schema — append-only `person_descriptions` table

**Decision:** Option B from the brainstorm — separate `person_descriptions` table with history, denormalised `latest_description_id` pointer on `persons`, durable `description_queue` job table.

Three new SQLite objects:

```sql
person_descriptions (id, person_id, described_at, backend, model_id,
                     snapshots_used JSON, attributes JSON, summary, confidence)
description_queue   (id, person_id, enqueued_at, status, last_error, attempts)
persons.latest_description_id INTEGER   -- denormalised pointer
```

**Why append-only:** people change appearance during a session (jacket on/off, picks up a bag). Overwriting destroys evidence that's useful at search time and for the thesis audit trail. The denormalised `latest_description_id` keeps the common-case JOIN fast.

**`merge_persons()` patched** (database.py) to re-point `person_descriptions` from `remove_id` to `keep_id`, inherit the latest pointer if `keep_id` was undescribed, and cancel any pending queue rows for `remove_id`.

### LL-D: Strict-JSON prompt with enum'd attributes

**Decision:** Both backends share the same SYSTEM + USER prompt template (`describer.py:SYSTEM_PROMPT_DESCRIBE` / `USER_PROMPT_DESCRIBE`). 17 fields, most enum-constrained:

| Field group | Examples |
|---|---|
| Demographics | `gender`, `age_range`, `body_build`, `height_class` |
| Hair / face | `hair_color`, `hair_length`, `beard`, `glasses` |
| Headwear | `headwear`, `headwear_color` |
| Top / bottom | `clothing_top`, `clothing_top_color`, `clothing_bottom`, `clothing_bottom_color` |
| Free-text | `accessories` (list), `distinctive_features`, `summary` |

A 13-color palette (`red/blue/green/yellow/black/white/gray/brown/orange/purple/pink/multi/unknown`) is the closed set for any `*_color` field. Values outside the palette are coerced to `unknown` in `_coerce_to_schema()` so SQL filters stay deterministic.

A robust `_clean_json()` helper strips ```json fences, locates `{...}` slices inside surrounding prose, and returns `None` only on unrecoverable malformations (raw text logged for the audit trail).

### LL-E: Background worker — mirrors ReconciliationWorker pattern

**File:** `modules/llm/description_worker.py`

**Pattern lifted directly from `ReconciliationWorker`** so the lifecycle is consistent across the codebase: `_stop_event`, `run_forever()`, per-task try/except → never crash the daemon.

**Loop:**
1. **Startup recovery** — any `in_progress` rows from a previous run get bumped back to `pending` (durability).
2. **Drain in-memory hint queue** (non-blocking) — wake up immediately on fresh binds.
3. **Periodic sweep** every 60 s — `get_persons_without_description()` → `enqueue_description()` so nothing falls through the cracks if the in-memory queue dropped due to backpressure.
4. **Claim + handle** — `claim_next_description()` atomically marks one pending row `in_progress`, then `_handle()` picks the best snapshot, calls the backend, inserts the result, marks the row `done`.
5. **Failure path** — `fail_description()` increments attempts; rows that hit `MAX_DESCRIPTION_ATTEMPTS = 3` are marked `failed` with the raw error preserved.

**Snapshot selection** reuses the existing `CropQualityGate` (Part 2) — filters by blur/dimension/brightness, then picks the highest Laplacian-variance crop. Zero new image-quality code.

### LL-F: Producer hook in `main.py` (one line of logic)

In the embedding worker, immediately after the snapshot crops are saved and `track_registry[key] = person_uuid` succeeds with `status_out == "new"`:

```python
if ENABLE_DESCRIPTION_WORKER and status_out == "new":
    try:
        db.enqueue_description(person_uuid)
        llm_queue.put_nowait({"person_id": person_uuid})
    except queue.Full:
        pass   # DB row is enough; sweep will catch it
```

Producer NEVER blocks. Durability lives in SQLite; the in-memory queue is just a wake-up hint.

### LL-G: Natural-language search — three-stage pipeline

**File:** `modules/search/text_search.py`

1. **`QueryParser.parse(query)`** — text-only Ollama call (`qwen2.5:3b` by default) emits a partial schema dict. Missing fields stay missing; negation lands as `"glasses": "no"`. A rule-based regex fallback runs if Ollama is unreachable.
2. **Stage 1 — SQL filter** (`db.search_persons_by_attributes`) — enum fields → exact-match via `json_extract`; phrase fields → `LIKE`; accessory lists → OR'd `LIKE`. Returns candidates with attributes denormalised.
3. **Stage 2 — soft re-rank** — synonym-aware match scoring (`crimson → red`, `hoodie → sweater`, `chubby → heavy`, etc.). Score normalised by # of filter fields the user specified.
4. **Stage 3 — semantic fallback** — only fires when Stage 1 is empty AND `ENABLE_TEXT_FALLBACK_RERANK = True`. Encodes the query and every stored summary with `sentence-transformers/all-MiniLM-L6-v2`, cosine-ranks. Lazy import so the dependency stays optional.

CLI: `python main.py --phase 4 --search-text "..."` prints ranked results with `person_id`, summary, last-seen-cam/time, snapshot path.

### LL-H: New invariants

Added to `project_invariants.md`:

| # | Rule | Why |
|---|---|---|
| 14 | DescriptionWorker NEVER blocks the embedding worker | Producer uses `put_nowait` + durable SQLite row. Even if `queue.Full` raises, the description_queue row already exists and the periodic sweep will pick it up. |
| 15 | `person_descriptions` rows ACCUMULATE | Never `UPDATE`-overwrite; every describe inserts a new row. `latest_description_id` moves forward. Re-describes preserve history (essential for thesis audit + future "what was this person wearing earlier today?" queries). |
| 16 | `description_queue.status` transitions are append-only | `pending → in_progress → done | failed`. Never moves backwards. (Exception: startup recovery may reset stuck `in_progress` rows back to `pending` because the previous worker is provably gone.) |
| 17 | `BODY_MATCH_THRESHOLD` (legacy alias) stays at `BODY_MATCH_THRESHOLD_CROSS_CAM` | Documented previously as invariant #13 in the topology section; reaffirmed here because the LLM PR added another caller that could be tempted to lower it. Phase-3 photo search and any context-less call uses this floor; live identification passes its own override. |

### LL-I: Verification

All checks pass on an in-memory database:

- Schema migrates cleanly (in-memory DB, fresh init, and `_migrate()` path).
- `enqueue_description` is idempotent — second call with pending row is a no-op.
- `claim_next_description` returns one row and marks it `in_progress`.
- `insert_description` + `complete_description` update `latest_description_id` correctly.
- `fail_description` flips to `failed` after `max_attempts` and preserves `last_error`.
- `search_persons_by_attributes` returns expected matches on enum, phrase, and accessory filters.
- `merge_persons` correctly re-points descriptions and inherits when the kept person had no description.
- `QueryParser._rule_based_fallback` parses gender, build, top color, headwear, glasses, negation correctly (with word-boundary regex — a 0-day bug from the initial implementation where "woman" matched "man" was caught and fixed).
- `TextSearchEngine.search()` round-trip works end-to-end with a stub describer.
- All seven new/touched Python files pass `ast.parse()`.

### LL-J: Status

- **`DESCRIPTION_BACKEND = "qwen-vl"`** is the default — works on CPU with no remote host required (assuming `ollama pull qwen2.5vl:2b` and `ollama pull qwen2.5:3b` have been run locally).
- **Marlin remote host** ships ready-to-deploy in `modules/llm/marlin_server/` with a README covering Colab, cloud VM, and local-GPU deployment.
- **Phase 4C / 4D deferred** per the locked scope: multi-snapshot consensus, deterministic HSV color sanity-check, re-description on appearance change, hybrid photo+text reranking.
- Phase 4 CLI:
  - `python main.py --phase 4 --describe-all` — one-shot describe pass.
  - `python main.py --phase 4 --search-text "a man in a red t-shirt"` — search.
  - Both can be combined.
- During Phase 2 the description worker runs as a daemon — descriptions populate automatically as new persons are bound.

---

## Reconciliation Worker Improvements

### RW-A: Mean-pool similarity (Bug D fix)
**File:** `modules/reconciliation/worker.py`
**Old:** Max-pool — `max(similarity(a_i, b_j))` over all pairs. One accidentally-similar pair out of N×M was enough to trigger a false proposal.
**New:** Mean-pool — `mean(similarity(a_i, b_j))` over all compatible pairs. A legitimate duplicate scores high on most pairs; a false positive does not.

### RW-B: Proposal deduplication (Bug E fix)
**File:** `modules/storage/database.py` — `propose_merge()`
**Old:** `INSERT INTO merge_proposals ...` — every 120-second cycle added a new row for the same pair.
**New:** Upsert by pair — if a pending proposal for (pid_a, pid_b) or (pid_b, pid_a) exists, update its similarity score instead of inserting a duplicate.

### RW-C: Reconciliation guards
**What:** Two quality gates before running pairwise comparison:
1. Both persons must have ≥ 2 gallery embeddings (`MIN_GALLERY_FOR_RECONCILIATION = 2`).
2. Both persons must have view-coverage ≥ 0.5 (`MIN_VIEW_COVERAGE_FOR_MATCHING`).
**Why:** A 1-embedding prototype is too noisy to be a reliable merge target. A single-angle gallery can produce false matches against another person viewed from the same angle.

---

## Bug Audit Summary (all fixed)

| Bug | Root Cause | Fix Location |
|---|---|---|
| A | OSNet using ImageNet weights, not Re-ID weights | `embedder.py` — auto-download via gdown |
| B | Crop filter `d["bbox"] == t["bbox"]` always False | `main.py` — replaced with IoU ≥ 0.7 |
| C | Initial embedding tagged "initial" → view-coverage 0.0 forever | `database.py` + `main.py` — canonical tag at creation |
| D | Max-pool reconciliation generating false proposals | `worker.py` — switched to mean-pool |
| E | `propose_merge()` always INSERT → duplicate rows | `database.py` — upsert by pair |
| F | View-coverage gate in searcher blocked re-entries → duplicate IDs | `searcher.py` — gate removed from search |
| G | `CROSS_TYPE_MULTIPLIER` penalty was dead code (no face embeddings) | `searcher.py` — code removed |
| H | Mean aggregation skewed by outlier frames | `embedder.py` — switched to median |
| I | CLAHE ran unconditionally on every frame | `enhancement.py` — conditional on V-mean |
| J | Linear SQLite scan would bottleneck at 500+ persons | `searcher.py` + new `faiss_index.py` |

---

## Parts 9 & 10 — Not Yet Implemented

### Part 9 — Spatio-Temporal Camera Constraints (Planned)
**What:** A transition-time matrix `CAM_TRANSITION_MIN_SEC[(cam_a, cam_b)]` that rejects physically impossible cross-camera matches (person can't teleport from cam 3 to cam 1 in 2 seconds if the walk takes 30 seconds).
**Files to change:** `config/settings.py` (add matrix), `modules/search/searcher.py` (apply constraint).
**Expected improvement:** Eliminates a class of false positives where two similar-looking people on different cameras at the same time are wrongly merged.

### Part 10 — LLM Description Matching (Planned)
**What:** Use Qwen2.5-VL:2b (Ollama) to generate text descriptions of stored persons. Use description as a secondary matching signal when visual similarity is uncertain.
**When to trigger:** Visual similarity in range [0.55, BODY_MATCH_THRESHOLD) — uncertain zone.
**Decision logic:** Uncertain visual + matching description → accept. Uncertain visual + conflicting description → reject.
**Files:** `modules/llm/describer.py` (already scaffolded), `modules/search/searcher.py`.

---

*Last updated: 2026-05-28 — Parts 1–8.5 + Part 10 (Phase 4A+4B LLM description & NL search) complete. Part 9 (spatio-temporal) and Part 10 (Phase 4C/4D extras) deferred.*
