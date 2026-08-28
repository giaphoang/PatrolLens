# PatrolLens

PatrolLens makes long body-worn camera video searchable with natural-language queries and returns verified timestamp intervals with supporting evidence.

The implementation follows one rule:

> Cheap local models find where to look → Gemini determines what happened → targeted re-inspection determines exactly when.

It is not frame pooling and it does not upload an entire 90-minute video for every query.

## Architecture

```mermaid
flowchart LR
    V["Bodycam videos<br/>up to 90 min"]

    subgraph OFF["1. Offline ingestion — once per video"]
        FF["FFmpeg / FFprobe"]
        CH["Scene/change detection<br/>keyframes + pHash dedup"]
        GE["OpenRouter<br/>Gemini Embedding 2<br/>unique images + ASR text · 768-d"]
        EXACT["OpenRouter Whisper<br/>exact searchable speech"]
        CLAP["larger_clap_general CoreML<br/>10 s audio · 512-d"]
        CUES["Optional Silero VAD<br/>speech-presence cues"]
        IDX["Timestamped evidence<br/>SQLite FTS5 + pgvector/FAISS"]
    end

    subgraph RET["2. Cheap query-time retrieval"]
        PLAN["Gemini query planner"]
        PAR["Parallel visual / ASR / CLAP audio search"]
        FUSE["Temporal join + weighted RRF"]
    end

    subgraph ACTIVE["3. Active perception"]
        AGENT["Gemini 3.1 Pro"]
        TOOLS["get_frames · get_audio · get_clip"]
        MEM["Durable evidence memory"]
    end

    VERIFY["4. Independent Gemini event verifier"]
    REFINE["5. High-resolution timestamp refinement"]
    TL["Optional TimeLens2 adapter"]
    OUT["video + [start,end]<br/>confidence + evidence"]

    V --> FF
    FF --> CH --> GE --> IDX
    FF --> EXACT --> IDX
    FF --> CLAP --> IDX
    FF --> CUES --> IDX
    Q["Investigator query"] --> PLAN --> PAR
    IDX --> PAR --> FUSE --> AGENT
    AGENT --> TOOLS --> MEM --> AGENT
    AGENT --> VERIFY --> REFINE --> OUT
    REFINE -. "broad or low-quality interval" .-> TL -.-> OUT
```

This resolves the original hard case—“the moment the person in the red jacket started shouting”—as three separate questions:

1. Does visual retrieval place a red-jacket person near this time?
2. Does audio retrieval place raised vocal intensity nearby?
3. After directly viewing and listening, can Gemini attribute the voice to that person and locate the onset?

Retrieval scores create candidates only. They never become final evidence confidence.

## Indexed evidence channels

| Component | Evidence it creates | Queries it directly supports |
|---|---|---|
| ASR | Timestamped spoken words | Miranda rights, quoted speech, names, commands |
| larger_clap_general (`full`) | Timestamped 512-d raw-audio semantics | Shouting, sirens, gunshots, barking, overlapping acoustic events |
| Silero VAD (`full`) | Timestamped speech presence | Speech/non-speech candidate filtering |
| Gemini Embedding 2 | Shared 768-d image/text semantic space | Clothing, vehicles, scenes, transcript meaning |
| Gemini clips | Motion and cross-modal association | Handcuffing, traffic-stop sequence, who shouted, event causality |

ASR cannot establish a visual action, and a CLAP match does not prove who made
the sound. Gemini verifies those cross-modal claims after coarse retrieval. New
ingestion does not run OCR or handcrafted loudness/pitch analysis; historical
rows from older indexes remain searchable.

## Quick start

Prerequisites: Python 3.11+ and FFmpeg/FFprobe.

Install the production PostgreSQL/OpenRouter stack, then install the paired
CoreML/ONNX CLAP runtime and model artifacts on Apple Silicon:

```bash
uv sync --extra dev --extra full
scripts/setup_clap_coreml_macos.sh
source .venv/bin/activate
patrol-lens doctor
```

The local SigLIP/FAISS and faster-whisper paths remain optional fallbacks and
are not included in `full` because production uses OpenRouter embeddings,
PostgreSQL, and OpenRouter transcription:

```bash
uv sync --extra full --extra vision --extra speech
```

Create the local environment file from the template:

```bash
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY.
```

The `patrol-lens` CLI automatically loads the nearest `.env` before parsing its settings. Values in that file take precedence over inherited shell variables, preventing an older exported `OPENROUTER_API_KEY` from being selected accidentally. It does not print or commit the key.

The default reasoning model is the OpenRouter slug `google/gemini-3.1-pro-preview`. Override it with `PATROLLENS_GEMINI_MODEL` or `--model`. The planner can use a separate model via `PATROLLENS_GEMINI_PLANNER_MODEL`. The optional `PATROLLENS_OPENROUTER_BASE_URL`, `PATROLLENS_OPENROUTER_HTTP_REFERER`, and `PATROLLENS_OPENROUTER_TITLE` settings are also supported.

PatrolLens uses `https://openrouter.ai/api/v1`; it does not use the direct
provider SDKs. Ingestion sends five-minute extracted WAV chunks to the dedicated
OpenRouter transcription endpoint using `PATROLLENS_ASR_MODEL`, then sends
deduplicated keyframe images and ASR text through the embeddings endpoint.
Raw video is sent only after coarse retrieval through bounded active-perception
tools.

Set the embedding model and vector size in the environment when resuming or rebuilding an index. Ingestion does not require a separate batch-model CLI option:

```bash
export PATROLLENS_EMBEDDING_MODEL=google/gemini-embedding-2
export PATROLLENS_EMBEDDING_DIMENSIONS=768
```

`PATROLLENS_EMBEDDING_QUERY_MODEL` may optionally override the model used for investigator queries. `PATROLLENS_EMBEDDING_BATCH_MODEL` is reserved for a future asynchronous text-only Batch API path; current ingestion does not use it because multimodal inputs are sent through the synchronous endpoint.

For an existing PostgreSQL index, run the idempotent 768-dimensional backfill before resuming ingestion:

```bash
set -a; source .env; set +a
patrol-lens migrate-embeddings \
  --backend postgres \
  --database-url "$PATROLLENS_DATABASE_URL" \
  --index .patrol-lens-artifacts
```

This re-embeds canonical transcript text, historical OCR text, and visual image evidence into `pl_embeddings.embedding_768`, creates the `pl_embeddings_embedding_768_hnsw` index, and leaves any legacy `embedding` vectors intact for audit. Legacy raw-video/audio rows are deliberately not re-embedded; optimized ingestion recreates visual evidence as deduplicated keyframes and, in the full profile, audio as Silero speech-presence evidence. Resume normal ingestion with the same environment and without `--force`.

### 1. Ingest videos

Core profile: Gemini Embedding 2 for deduplicated keyframe and ASR text
indexing plus OpenRouter `openai/whisper-large-v3-turbo` segment transcripts:

```bash
patrol-lens ingest videos_corpus --index .patrol-lens --profile core
```

Full profile adds local larger_clap_general CoreML audio embeddings and Silero
VAD speech-presence evidence. CLAP decodes the original video directly at 48
kHz; it never uses Whisper's 16 kHz WAV:

```bash
patrol-lens ingest videos_corpus --index .patrol-lens --profile full
```

Use `--clap` to add CLAP to the core profile or `--no-clap` to disable it in
the full profile. CLAP is fixed at a 10-second window; the default 5-second
stride can be changed with `--clap-stride-s`.

`PATROLLENS_CLAP_COMPUTE_UNITS=cpu_only` is the safe default on this macOS 26
M3 environment and measured about 21 ms steady-state per window. The downloaded
artifact advertises `cpu_and_gpu`, but Apple's MPSGraph compiler aborts on this
host; use `--clap-compute-units cpu_and_gpu` only after a canary succeeds on the
target OS/runtime.

`--no-embedding-images` disables hosted keyframe vectors while retaining ASR
text embeddings. `--no-embeddings` selects the local SigLIP2 visual path and
disables all hosted ingestion vectors. There is no ingestion option for
raw-video embeddings.

For a dependency-free metadata smoke test:

```bash
patrol-lens ingest videos_corpus --index .patrol-lens-smoke --profile metadata
```

Ingestion is fingerprinted and restartable. Re-run the same command against the same `--index` to continue: completed videos are skipped, while failed or incomplete videos are retried. Each provider response is validated, saved immediately in a content-hash cache, and then attached to evidence in small committed batches. A retry reuses cached vectors instead of paying to embed completed items again. Do not use `--force` when continuing. Changing the embedding model, dimensions, or extraction settings creates a new fingerprint; use `--force` only for an intentional evidence rebuild (cached vectors are still reused when their full cache key matches).

When CLAP is first enabled on an existing completed corpus, ingestion performs
an additive CLAP-only backfill. Existing transcripts, visual vectors, and
evidence are preserved. Every 10-second audio vector is checkpointed before the
next window, so an interruption resumes from uncached windows without rerunning
the completed audio model calls.

Changing the ASR backend does not reprocess completed videos. On an incomplete
video, existing transcript evidence is reused and the remaining modalities
continue. To deliberately regenerate transcripts, use `--force`; this also
rebuilds the rest of that video's evidence.

Before switching a corpus, benchmark one representative canary:

```bash
patrol-lens benchmark-asr videos_corpus/canary.mp4
```

The JSON report includes wall time, real-time factor, provider cost, cache
usage, and a transcript preview. Add `--reference-transcript expected.txt` to
measure word error rate against a known transcript. This affects offline
ingestion only; query-time verification and temporal refinement remain
unchanged.

## Traceable PostgreSQL + pgvector backend

For an auditable deployment, use PostgreSQL instead of the default SQLite/FAISS development backend:

```bash
docker compose -f compose.pgvector.yaml up -d
uv sync --extra postgres
export PATROLLENS_DATABASE_URL='postgresql://patrol_lens:patrol_lens@localhost:5435/patrol_lens'

patrol-lens ingest videos_corpus \
  --backend postgres \
  --database-url "$PATROLLENS_DATABASE_URL" \
  --index .patrol-lens-artifacts \
  --profile full
```

Use the same `--backend postgres --database-url ...` flags for `retrieve`, `search`, and `evaluate`. `--index` remains the local artifact root for extracted frames, audio, and agent memory; the PostgreSQL DSN is only for the canonical evidence/index database.

```mermaid
flowchart LR
    RAW["Raw video"] --> LOCAL["Local scene/keyframe dedup\nASR · CLAP · optional VAD"]
    LOCAL --> E["pl_evidence\ncontent · timestamps · metadata\nevidence_hash"]
    E --> GE["Gemini Embedding 2\nunique images + ASR text\n768-d only"]
    E --> CE["larger_clap_general\nraw audio · 512-d only"]
    GE --> CACHE["pl_embedding_cache\ncontent-addressed checkpoint"]
    CE --> ACACHE["pl_audio_embedding_cache\nwindow checkpoint"]
    E --> TX["One ACID transaction"]
    CACHE --> TX
    ACACHE --> TX
    TX --> V["pl_embeddings\npgvector + duplicated provenance\nFK evidence_id"]
    TX --> AV["pl_audio_embeddings\nvector(512) + provenance\nFK evidence_id"]
    Q["Query embedding"] --> PG["pgvector cosine search"]
    PG --> ROW["PostgreSQL embedding row"]
    ROW --> REPLAY["source_uri + [start,end]\nevidence + source hashes\nreplay / verify"]
```

Every PostgreSQL embedding is inserted with `INSERT ... SELECT` from its canonical evidence and asset rows. Gemini vectors use `pl_embeddings.embedding_768`; CLAP uses the separate `pl_audio_embeddings.embedding_512` HNSW index. Both rows carry replayable provenance and are guarded by foreign keys and provenance triggers. The two semantic spaces are never compared directly; weighted RRF combines their ranked hits.

### 2. Inspect coarse candidates

```bash
patrol-lens retrieve \
  "Find the moment the person in the red jacket started shouting" \
  --index .patrol-lens \
  --planner heuristic
```

Use `--planner gemini` for semantic query decomposition. The heuristic planner is useful for local debugging and evaluation.

### 3. Run the complete search lifecycle

```bash
patrol-lens search \
  "Find the moment the person in the red jacket started shouting" \
  --index .patrol-lens \
  --model google/gemini-3.1-pro-preview
```

Other examples:

```bash
patrol-lens search "Find all instances of a vehicle being pulled over at night"
patrol-lens search "Find every moment where someone raises their voice"
patrol-lens search "Locate all footage containing a person in a red shirt"
patrol-lens search "Find all interactions where an officer reads Miranda rights"
patrol-lens search "Find all license plates visible and tell me what each says"
patrol-lens search "Find moments where a suspect is being handcuffed"
```

Each accepted result contains:

```json
{
  "video_id": "video-…",
  "video_path": "/absolute/path/bodycam.mp4",
  "start_ms": 1877200,
  "end_ms": 1883800,
  "confidence": 0.87,
  "description": "The red-jacket person begins shouting",
  "evidence": {
    "visual": ["red jacket remains on the speaking subject"],
    "audio": ["voice intensity rises at approximately 1877.2 s"],
    "transcript": ["get away from me"],
    "ocr": []
  },
  "grounding_method": "gemini_lightweight"
}
```

## How a 90-minute video is handled

At the defaults, one 90-minute video retains 674 overlapping 16-second temporal windows at an 8-second stride for joining and recall, samples about 5,400 frames at 1 FPS, and produces 1,079 CLAP windows at 10 seconds / 5-second stride. Temporal windows are never uploaded for embedding. CLAP streams 48 kHz mono float32 directly from the original video and retains only the overlap buffer; it does not write another full-length WAV. The separate 16 kHz extraction remains for OpenRouter transcription and optional Silero VAD.

At query time, exact FTS5/Postgres text search is retained for literal ASR
strings and any historical evidence rows. Gemini Embedding 2 query vectors
search semantic transcript/visual evidence, while the paired CLAP ONNX text
encoder searches only the 512-d raw-audio index. FAISS or pgvector
reduces the full corpus to roughly 5–20 candidate intervals. Gemini receives
only bounded observations—at most 12 frames, 30 seconds of audio, or 20 seconds
of video per action—with a five-turn default budget. Memory grows as compact
JSON summaries, not raw 90-minute media.

See [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) for the formalization and detailed lifecycle.

## Optional TimeLens2

TimeLens2 is after verification, never in candidate generation. It activates only when a supported interval is still broad or low confidence.

PatrolLens expects an independently installed wrapper command that accepts:

```text
--video PATH --query TEXT --start-ms N --end-ms N
```

and prints:

```json
{"intervals_ms": [[12000, 15400, 0.82]]}
```

Enable it only after reviewing its upstream terms:

```bash
patrol-lens search "..." \
  --timelens-command "python /path/to/timelens2_wrapper.py" \
  --acknowledge-timelens-license
```

The inspected TimeLens2 top-level license limits use to academic purposes and excludes EU use. It is therefore not installed, imported, or enabled by default. See [THIRD_PARTY.md](THIRD_PARTY.md).

## Evaluation

Coarse-retrieval labels use JSONL:

```json
{"query":"Miranda rights","relevant":[["video-id",120000,145000]]}
```

Run:

```bash
patrol-lens evaluate evaluation.jsonl --index .patrol-lens --planner heuristic
```

The evaluator reports recall@K and mean best temporal IoU. Production evaluation should additionally label verifier precision, start/end error, evidence attribution, and false-positive severity.

## Tests

```bash
uv run --extra dev pytest -q
```

Tests cover normalized storage, independent 768-d/512-d guards, CLAP windowing and checkpoint recovery, additive audio backfill, keyframe deduplication, FTS/vector retrieval, multimodal temporal joining, active-perception control, restartable ingestion, TimeLens2 gating, CLI contracts, and real FFmpeg extraction.

## Privacy boundary

Raw corpus video, extracted transcripts, indexes, and agent memory remain local.
Offline indexing sends deduplicated keyframe images, transcript text, and
five-minute audio chunks to OpenRouter. Query-time search sends the investigator
query, and active search sends only selected short observations after coarse
retrieval. CLAP audio and text inference remains local on the M3. Deployments
still need agency-specific access control, encryption,
retention, redaction, audit logging, provider data-handling review, and legal
review; those are intentionally outside this prototype.
