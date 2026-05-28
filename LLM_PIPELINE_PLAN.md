# Plan — Part 10 / Phase 4A + 4B: LLM Body Description + Natural-Language Search

## Context

Body re-identification (Parts 1–8.5) gives us a stable `person_id` and a multi-view gallery for every person. That makes "find the same person again" work, but not "find a man in a red t-shirt and white hat" — the system has no semantic understanding of what people look like.

**This delivery adds two capabilities, both on top of the existing pipeline:**

1. **Body description (Phase 4A).** For every confirmed person, generate a structured description of their appearance from their best snapshots, in a strict JSON schema with enum'd attributes. Run it on a background daemon that mirrors the existing `ReconciliationWorker` pattern so detection/tracking/embedding are never blocked.

2. **Natural-language search (Phase 4B).** Accept a free-text query ("a fat man with a red t-shirt and white hat"), parse it into the same structured schema with a small text-only LLM, run a two-stage retrieval (SQL filter → soft re-rank), return ranked persons with snapshots.

This is the body track only. Face description/recognition is the teammates' track on the same `person_id` namespace — both pipelines will compose later.

**Decisions locked from brainstorming:**

| Question | Choice |
|---|---|
| VLM | **Marlin-2B as primary** (user preference); Qwen2.5-VL 2B as local-CPU fallback. Abstract describer interface so either can drive the worker. |
| Schema | **Option B — separate `person_descriptions` table** with one row per description version (history kept). |
| Scope | **4A + 4B MVP first**: describer + worker + parser + search. 4C/4D (multi-snapshot consensus, color sanity-check, re-description triggers) deferred. |
| Query parser | **Small text-only LLM** (`qwen2.5:3b` via Ollama). Different model from the describer; cheap and fast. |

**⚠️ Marlin-2B constraint reminder.** Marlin-2B is a video VLM that requires a GPU (per its model card — H100 training target, no CPU-only support, transformers stack only, no Ollama). SURVEILLANT's local target is CPU-only. We handle this with the abstract describer interface and a **remote HTTP backend** for Marlin: you run Marlin on a GPU host (Colab notebook, cloud VM, or any machine with a GPU) exposing a small FastAPI endpoint; SURVEILLANT POSTs snapshots over HTTP. The local Qwen2.5-VL backend always works without a GPU and is the default fallback if Marlin's host is unreachable.

---

## Architecture

```
                                       ┌─────────────────────────────┐
                                       │ Marlin-2B host (remote GPU) │
                                       │  POST /describe → JSON      │
                                       └─────────────▲───────────────┘
                                                     │ HTTP
   ┌─────────────────────┐  ┌──────────────────────┐ │  ┌─────────────────────┐
   │ embedding_worker    │  │ DescriptionWorker    │─┘  │ search CLI / API    │
   │ (existing)          │→ │ (NEW, daemon)        │    │ (NEW)               │
   │ binds person_id     │  │ pulls llm_queue,     │    │ parse query (text   │
   │ → enqueue describe  │  │ picks best snapshot, │    │  LLM) → SQL filter  │
   │                     │  │ calls Describer ABC, │    │  → soft rerank      │
   │                     │  │ writes results       │    │                     │
   └─────────┬───────────┘  └──────────┬───────────┘    └─────────┬───────────┘
             │                         │                          │
             ▼                         ▼                          ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                          SQLite (source of truth)                       │
   │ persons (+ latest_description_id ptr) • person_descriptions (NEW)       │
   │ description_queue (NEW, durable pending list)                           │
   └─────────────────────────────────────────────────────────────────────────┘
```

**Producer.** The embedding worker's identify branch (`main.py` ~lines 624–631, right after `track_registry[key] = person_uuid`) enqueues `("describe", person_id)` on `llm_queue` after a successful person-bind. It also writes to `description_queue` (durable SQLite table) so restart-survival is automatic.

**Consumer.** `DescriptionWorker` daemon thread (same pattern as `ReconciliationWorker`) pulls from `llm_queue`, loads pending tasks from `description_queue` on startup, dispatches to the configured `Describer` backend, writes structured output to `person_descriptions`, updates the durable queue row to `done`.

**Backends behind one ABC.**
```python
class Describer(ABC):
    def describe(self, snapshot_paths: list[str]) -> dict | None: ...

class MarlinRemoteDescriber(Describer):     # HTTP POST to GPU host
class QwenVLOllamaDescriber(Describer):     # local Ollama, CPU
```

Settings drive selection: `DESCRIPTION_BACKEND = "marlin" | "qwen-vl"`; Marlin-only constants `MARLIN_HOST = "http://gpu-host:8000"` and timeouts.

**Query side.** `QueryParser` calls `qwen2.5:3b` text-only via Ollama with the parsing prompt (§5), returns a dict in the same schema. The retrieval layer runs Stage 1 SQL filter then Stage 2 soft re-rank on the candidate set.

---

## Schema (one new table, one new column, one durable queue)

In `surveillant/modules/storage/database.py`, add to `_init_schema` and `_migrate`:

```sql
-- One row per description version. Re-describes ACCUMULATE; never overwrite.
CREATE TABLE IF NOT EXISTS person_descriptions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id       TEXT NOT NULL,
    described_at    TEXT NOT NULL,
    backend         TEXT NOT NULL,         -- "marlin" | "qwen-vl"
    model_id        TEXT NOT NULL,         -- e.g. "NemoStation/Marlin-2B" or "qwen2.5vl:2b"
    snapshots_used  TEXT NOT NULL,         -- JSON list of snapshot paths
    attributes      TEXT NOT NULL,         -- JSON dict matching §4 schema
    summary         TEXT,
    confidence      REAL,
    FOREIGN KEY (person_id) REFERENCES persons(person_id)
);
CREATE INDEX IF NOT EXISTS idx_desc_pid ON person_descriptions(person_id);

-- Denormalised "latest" pointer for fast joins from persons → current attributes
ALTER TABLE persons ADD COLUMN latest_description_id INTEGER;

-- Durable pending queue (survives restarts; thesis-friendly audit trail)
CREATE TABLE IF NOT EXISTS description_queue (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id    TEXT NOT NULL,
    enqueued_at  TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending',  -- 'pending'|'in_progress'|'done'|'failed'
    last_error   TEXT,
    attempts     INTEGER DEFAULT 0,
    FOREIGN KEY (person_id) REFERENCES persons(person_id)
);
CREATE INDEX IF NOT EXISTS idx_descq_status ON description_queue(status);
```

**New `Database` methods** (mirroring existing API style):
- `enqueue_description(person_id)` — upsert (don't duplicate) into `description_queue`.
- `claim_next_description()` → row or None; atomically marks `in_progress`.
- `complete_description(queue_id, description_id)` — sets `done`, updates `latest_description_id` on `persons`.
- `fail_description(queue_id, error_msg)` — increments `attempts`, sets `failed` if `attempts >= MAX`.
- `insert_description(person_id, backend, model_id, snapshots_used, attributes, summary, confidence)` → description_id.
- `get_pending_descriptions(limit)` — for the worker's startup-recovery loop.
- `search_persons_by_attributes(filters: dict)` — Stage 1 SQL filter; returns `(person_id, latest_attributes)` candidate list.

---

## Snapshot selection (the "best crop" step)

Reuse existing infrastructure — no new ML:

1. Pull `snapshot_paths` from the person row.
2. Run each through `CropQualityGate` (`modules/preprocessing/quality_gate.py`) — drop blur/dark/tiny.
3. Score remaining crops by **Laplacian variance** (sharpness) — already used by the quality gate, no new code.
4. Pick the top-1 sharpest. (4A keeps it single-image; multi-image consensus is deferred to 4C.)

If the person has no quality-passing snapshot, mark the queue row `failed` with `last_error = "no usable snapshot"` — periodic sweep will re-try later (gallery may have grown).

---

## Prompts (§4 in the brainstorm — final form)

### 4.1 Description prompt (image → strict JSON)

Used by **both** backends (Marlin and Qwen2.5-VL). Same prompt, same schema, model-agnostic.

**System:** "You are a surveillance description assistant. Describe ONLY what you can see. Do not infer beyond the image. Use the value 'unknown' when you are not sure. Output ONLY a single JSON object, no commentary, no markdown fences."

**User (image + this template):**
```
Output JSON with EXACTLY these fields:
{
  "gender": "male" | "female" | "unknown",
  "age_range": "child" | "teen" | "young_adult" | "adult" | "older_adult" | "unknown",
  "body_build": "slim" | "average" | "heavy" | "unknown",
  "height_class": "short" | "average" | "tall" | "unknown",
  "hair_color": "black" | "brown" | "blonde" | "red" | "gray" | "white" | "unknown",
  "hair_length": "short" | "medium" | "long" | "bald" | "unknown",
  "beard": "yes" | "no" | "unknown",
  "glasses": "yes" | "no" | "unknown",
  "headwear": "<one short phrase or 'none'>",
  "headwear_color": "<color name from the palette below or 'unknown'>",
  "clothing_top": "<one short phrase: t-shirt|jacket|hoodie|shirt|coat|sweater|...>",
  "clothing_top_color": "<color from palette>",
  "clothing_bottom": "<jeans|shorts|skirt|pants|dress|...>",
  "clothing_bottom_color": "<color from palette>",
  "accessories": [<short phrases: "backpack" | "handbag" | "umbrella" | "watch" | ...>],
  "distinctive_features": "<one sentence or 'none'>",
  "summary": "<one human-readable sentence>"
}
Palette = red|blue|green|yellow|black|white|gray|brown|orange|purple|pink|multi|unknown
```

### 4.2 Query parsing prompt (text → partial JSON)

**System:** "You parse a security operator's spoken description into a search filter. Output ONLY JSON, no commentary. Omit any field the operator did not mention. Use 'no' for negations like 'without glasses'."

**User:** the raw query string.

Output is the same schema as §4.1 but with **missing fields meaning "don't filter on this"**. Negation handled by emitting the explicit value (`"glasses": "no"`).

### 4.3 JSON cleaning

Both prompts: describer's response is run through a small `_clean_json(raw: str)` function — strips ```json fences, balances braces, returns `None` on irrecoverable malformations. Logs the raw text on failure for the thesis audit trail.

---

## Search retrieval (the user-facing magic)

**Stage 1 — Structured SQL filter.** From the parsed query dict, build a parameterised WHERE clause against `persons` joined to its `latest_description_id` row in `person_descriptions`. Enum fields → exact match. Phrase fields (`clothing_top`, `accessories`) → `LIKE` match on attributes JSON via `json_extract()` (SQLite has this built-in).

**Stage 2 — Soft re-rank.** For each candidate, compute:
- `score = 0`
- `+1.0` per enum field that matches
- `+0.5` per phrase field with a synonym match (small static synonym table for colors and common garments — `"crimson" → "red"`, `"hoodie" → "sweater"`, etc.)
- Normalise by the number of fields the query specified.

Rank by score; return top-K with `person_id`, snapshot path, summary, score, last-seen-cam/time.

**Stage 3 — Fallback if Stage 1 returns empty.** Encode the parsed query as a sentence with `sentence-transformers/all-MiniLM-L6-v2` (22M params, CPU-fast); cosine-rank all `person_descriptions.summary` rows by it. Returns top-K. Catches edge cases where the user's phrasing didn't enum-match anything.

---

## The DescriptionWorker — concrete shape

New file: `surveillant/modules/llm/description_worker.py`. Pattern lifted from `ReconciliationWorker`:

```python
class DescriptionWorker:
    def __init__(self, describer: Describer, db, llm_queue, llm_pending, llm_pending_lock):
        self._stop_event = threading.Event()
        self._describer = describer
        self._db = db
        self._q = llm_queue
        self._pending = llm_pending
        self._lock = llm_pending_lock

    def run_forever(self):
        # 1. Startup recovery: load 'pending' and 'in_progress' rows from
        #    description_queue, re-enqueue them.
        # 2. Loop: pop from llm_queue (timeout 1s),
        #    claim_next_description() to acquire row lock,
        #    pick best snapshot, call describer.describe(),
        #    insert_description(), complete_description(),
        #    on exception → fail_description() and continue.
        # 3. Every K minutes do a sweep: any persons.latest_description_id IS NULL
        #    → enqueue (catches anything missed during high load).

    def stop(self): self._stop_event.set()
```

**Wired in `main.py` `run_phase2`:** instantiate describer per `DESCRIPTION_BACKEND` setting, instantiate worker, start daemon. Producer hook at the existing `track_registry[key] = person_uuid` point — single line: `db.enqueue_description(person_uuid); llm_queue.put_nowait({"person_id": person_uuid})`.

**Backpressure & safety:**
- Queue is bounded (`maxsize=200`). If full → drop the in-memory put, but the SQLite `description_queue` row is already written, so the periodic sweep still picks it up.
- Per-task try/except → never crash the daemon.
- Marlin HTTP timeouts → mark failed, sweep retries.
- Ollama unreachable → same.
- `MAX_DESCRIPTION_ATTEMPTS = 3` to stop retry storms.

---

## Critical files to modify / create

| File | Change |
|---|---|
| **NEW** `surveillant/modules/llm/describer.py` | Implement `Describer` ABC, `MarlinRemoteDescriber` (HTTP), `QwenVLOllamaDescriber` (Ollama), `QueryParser` (text-only Ollama), `_clean_json()` helper. Drop the existing `NotImplementedError` stub. |
| **NEW** `surveillant/modules/llm/description_worker.py` | `DescriptionWorker` daemon. Mirrors `ReconciliationWorker`. |
| **NEW** `surveillant/modules/llm/marlin_server/` | Small FastAPI script + README for the remote GPU host. Single endpoint `POST /describe` that takes images + returns JSON. Optional but ships with the delivery so the user can deploy it cleanly on Colab/cloud. |
| **NEW** `surveillant/modules/search/text_search.py` | `TextSearchEngine` class: parse → Stage 1 SQL filter → Stage 2 soft rerank → Stage 3 fallback. |
| `surveillant/modules/storage/database.py` | Add `person_descriptions` table, `description_queue` table, `latest_description_id` column to `persons`. Add the new methods listed in the Schema section. Update `_migrate()`. |
| `surveillant/config/settings.py` | Add `DESCRIPTION_BACKEND`, `MARLIN_HOST`, `MARLIN_TIMEOUT_SEC`, `OLLAMA_VLM_MODEL = "qwen2.5vl:2b"` (replaces `LLM_MODEL` for the VLM role), `OLLAMA_QUERY_MODEL = "qwen2.5:3b"`, `ENABLE_DESCRIPTION_WORKER = True`, `DESCRIPTION_SWEEP_INTERVAL_SEC = 60`, `MAX_DESCRIPTION_ATTEMPTS = 3`, `DESCRIPTION_QUEUE_MAXSIZE = 200`. |
| `surveillant/main.py` | (a) Imports for the new modules. (b) In `run_phase2`: build `llm_queue`, instantiate `Describer` per backend setting, start `DescriptionWorker` daemon, add producer line at the bind-success site. (c) New phase-4 CLI handler: `--describe-all` triggers a one-shot pass over all persons; `--search-text "<query>"` runs `TextSearchEngine.search()` and prints results. (d) Phase-dispatch block at the bottom gains an `elif args.phase == 4:` branch. |
| `DECISION_LOG.md` | New section "Part 10 — LLM Body Description + NL Search (Phase 4A + 4B)". What/Why/Old/New/Result. Document the Marlin-as-remote-backend design and the Qwen2.5-VL fallback. |
| `SURVEILLANT_ARCHITECTURE_REPORT.md` | Update the threading-model section (now 4 worker domains: detection, embedding, reconciliation, **description**). Update the schema snapshot. Update the mermaid diagram with the description loop. Add Phase 4 row to the phase history. |
| Memory: `project_surveillant.md` | Bump status table: Part 10 (4A + 4B) ✅; remaining items 4C/4D + Part 9 listed as next. |
| Memory: `project_invariants.md` | New invariants: (a) description worker NEVER blocks the embedding worker. (b) `person_descriptions` rows ACCUMULATE — never UPDATE-overwrite (re-describes insert new rows). (c) `description_queue` row state transitions are append-only (status moves forward, never back). |

---

## Existing utilities to reuse

- **`CropQualityGate`** (`modules/preprocessing/quality_gate.py`) — already filters by blur, dimensions, brightness. Used here to pick the best snapshot. No new image-quality code.
- **Snapshot paths in `persons.snapshot_paths`** — already a JSON list. Read directly; no schema change for snapshots.
- **`ReconciliationWorker`** pattern — `DescriptionWorker` is a direct copy with `Describer.describe()` swapped for `_mean_pool_similarity()`.
- **`embed_queue` + `pending_ids` + `pending_lock`** pattern in `main.py:285–290` — `llm_queue` + `llm_pending` + `llm_pending_lock` follow the same shape.
- **`Database._migrate()`** — already exists and handles ALTER TABLE additions for existing DBs. New columns and tables go through it.
- **JSON storage convention** — `snapshot_paths` is already stored as JSON text in SQLite (`json.dumps()` / `json.loads()`). `attributes` follows the same pattern.
- **`db.merge_persons(keep, drop)`** — when a future reconciliation merge fires, all rows in `person_descriptions` for `drop` get re-pointed to `keep` via a one-line UPDATE inside the existing merge transaction. Pre-existing transaction-commit invariant (#6) means this is safe.

---

## Verification

1. **Backend smoke tests.** Each `Describer` implementation has a `tests/test_describer_<backend>.py`:
   - `MarlinRemoteDescriber.describe()` against a known-up Marlin host returns a JSON dict with all required keys.
   - `QwenVLOllamaDescriber.describe()` against local Ollama (must have `qwen2.5vl:2b` pulled) returns a JSON dict with all required keys.
   - Both return `None` (not raise) when the backend is unreachable.
   - JSON cleaner survives ```json fenced output, leading/trailing prose, and extra commas.

2. **Schema migration.** Delete `surveillant/database/surveillant.db`. Run phase 2. Confirm `person_descriptions`, `description_queue`, and `persons.latest_description_id` exist via `sqlite3 .schema`.

3. **End-to-end 4A.**
   - Run phase 2 for ~3 minutes on the WiseNet videos.
   - Verify console shows `[DESCRIBE] person <uuid> ← {summary}` lines as the worker progresses.
   - Query DB: `SELECT person_id, summary FROM persons p JOIN person_descriptions d ON p.latest_description_id = d.id LIMIT 20;` — every confirmed person should have a row.
   - Open one snapshot manually, eyeball the description for plausibility.

4. **End-to-end 4B.**
   - `python main.py --phase 4 --search-text "a man in a red t-shirt"` → returns top-K persons with attribute match, ranked.
   - Negation: `--search-text "woman without glasses"` → query parser emits `glasses: "no"`, filter applies.
   - Empty Stage 1 fallback: query a description not in the DB → Stage 3 returns nearest summaries by embedding.

5. **Robustness.**
   - Stop Ollama mid-run → worker logs `[DESCRIBE FAILED]` and continues, never crashes. Restart Ollama → sweep picks up failed rows on next pass.
   - Stop Marlin host mid-run (if Marlin is the backend) → same behaviour.
   - Force malformed JSON (mock backend) → `_clean_json` either repairs it or returns `None` and the row is marked failed with the raw output preserved.

6. **Thread safety.**
   - `description_queue.claim_next_description()` is atomic (single UPDATE … WHERE status='pending' RETURNING) — no two workers claim the same row.
   - DescriptionWorker NEVER touches `track_registry` directly; only `db.*` calls.

7. **Performance budget.**
   - Verify embedding worker latency stays in its previous range — `[BELOW]`/`[MATCH]` log lines should not slow down measurably with the description worker running.
   - Measure end-to-end describe time per person and log it in the queue row for thesis evidence.

8. **Dashboard sanity.** Open `debug_dashboard.py` (existing) — `latest_description_id` joins should render the summary alongside the snapshot for each person. No code change required if the dashboard reads all `persons` columns generically; a small template tweak if not.