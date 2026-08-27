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
        LOCAL["SigLIP2 · faster-whisper<br/>PaddleOCR · Silero/RMS/pitch · YAMNet"]
        IDX["Timestamped evidence<br/>SQLite FTS5 + FAISS"]
    end

    subgraph RET["2. Cheap query-time retrieval"]
        PLAN["Gemini query planner"]
        PAR["Parallel visual / ASR / OCR / audio search"]
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

    V --> FF --> LOCAL --> IDX
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

## Why ASR and OCR exist

| Component | Evidence it creates | Queries it directly supports |
|---|---|---|
| ASR | Timestamped spoken words | Miranda rights, quoted speech, names, commands |
| OCR | Timestamped visible characters | License plates, signs, badge or unit numbers |
| Audio analysis | VAD, loudness, pitch, AudioSet events | Raised voice, sirens, gunshots, barking |
| Visual embeddings | Open-vocabulary frame similarity | Clothing, vehicles, nighttime scenes, visible objects |
| Gemini clips | Motion and cross-modal association | Handcuffing, traffic-stop sequence, who shouted, event causality |

ASR cannot read a license plate, OCR cannot transcribe Miranda rights, and neither can establish a visual action. Each modality has a narrow job.

## Quick start

Prerequisites: Python 3.11+ and FFmpeg/FFprobe.

Install the practical local stack plus the OpenAI SDK used to reach Gemini through OpenRouter:

```bash
uv sync --extra dev --extra openrouter --extra vision --extra speech --extra ocr
source .venv/bin/activate
patrol-lens doctor
```

For Silero VAD and YAMNet as well:

```bash
uv sync --extra dev --extra full
```

Set an OpenRouter key:

```bash
export OPENROUTER_API_KEY="..."
```

The default model is the OpenRouter slug `google/gemini-3.1-pro-preview`. Override it with `PATROLLENS_GEMINI_MODEL` or `--model`. The planner can use a separate model via `PATROLLENS_GEMINI_PLANNER_MODEL`. The optional `PATROLLENS_OPENROUTER_BASE_URL`, `PATROLLENS_OPENROUTER_HTTP_REFERER`, and `PATROLLENS_OPENROUTER_TITLE` settings are also supported.

PatrolLens uses the OpenAI Python SDK with `https://openrouter.ai/api/v1`; it does not use the direct Google SDK. The provider-prefixed model slug is passed unchanged to OpenRouter. Local frame, clip, and audio observations are encoded as the API's `image_url`, `video_url`, and `input_audio` content parts.

### 1. Ingest videos

Core profile: SigLIP2, faster-whisper `large-v3-turbo`, PaddleOCR, and local RMS/pitch analysis:

```bash
patrol-lens ingest videos_corpus --index .patrol-lens --profile core
```

Full profile also enables Silero VAD and YAMNet:

```bash
patrol-lens ingest videos_corpus --index .patrol-lens --profile full
```

For a dependency-free metadata smoke test:

```bash
patrol-lens ingest videos_corpus --index .patrol-lens-smoke --profile metadata
```

Ingestion is fingerprinted and restartable. Re-running the same configuration skips completed videos. Use `--force` to rebuild evidence for matching assets.

## Traceable PostgreSQL + pgvector backend

For an auditable deployment, use PostgreSQL instead of the default SQLite/FAISS development backend:

```bash
docker compose -f compose.pgvector.yaml up -d
uv sync --extra postgres
export PATROLLENS_DATABASE_URL='postgresql://patrol_lens:patrol_lens@localhost:5432/patrol_lens'

patrol-lens ingest videos_corpus \
  --backend postgres \
  --database-url "$PATROLLENS_DATABASE_URL" \
  --index .patrol-lens-artifacts \
  --profile core
```

Use the same `--backend postgres --database-url ...` flags for `retrieve`, `search`, and `evaluate`. `--index` remains the local artifact root for extracted frames, audio, and agent memory; the PostgreSQL DSN is only for the canonical evidence/index database.

```mermaid
flowchart LR
    RAW["Raw video + extracted media"] --> E["pl_evidence\ncontent · timestamps · metadata\nevidence_hash"]
    E --> TX["One ACID transaction"]
    TX --> V["pl_embeddings\npgvector + duplicated provenance\nFK evidence_id"]
    Q["Query embedding"] --> PG["pgvector cosine search"]
    PG --> ROW["PostgreSQL embedding row"]
    ROW --> REPLAY["source_uri + [start,end]\nevidence + source hashes\nreplay / verify"]
```

Every PostgreSQL embedding is inserted with `INSERT ... SELECT` from its canonical evidence and asset rows. The `pl_embeddings` row stores `evidence_id`, `video_id`, timestamps, modality, `source_uri`, evidence text/metadata, model version, confidence, `evidence_hash`, `source_sha256`, and `embedding_hash`. A foreign key, provenance trigger, and transactional ingestion path prevent orphan or mismatched embeddings. PostgreSQL WAL/PITR then covers both the vector and its audit metadata together.

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

At the defaults, one 90-minute video produces 674 overlapping 16-second processing windows at an 8-second stride and about 5,400 sampled frames at 1 FPS. The audio track is decoded once. Evidence is stored by its true timestamp rather than duplicated as opaque window documents.

At query time, FTS5 and FAISS reduce the full corpus to roughly 5–20 candidate intervals. Gemini receives only bounded observations—at most 12 frames, 30 seconds of audio, or 20 seconds of video per action—with a six-turn default budget. Memory grows as compact JSON summaries, not raw 90-minute media.

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

Tests cover normalized storage, FTS/vector retrieval, multimodal temporal joining, active-perception control, 90-minute windowing, restartable ingestion, TimeLens2 gating, CLI contracts, and real FFmpeg extraction.

## Privacy boundary

Raw corpus video, extracted ASR/OCR, indexes, and agent memory remain local. When full search is used, only the investigator query and selected short media observations are sent through OpenRouter to the configured Gemini model. Deployments still need agency-specific access control, encryption, retention, redaction, audit logging, and legal review; those are intentionally outside this prototype.
