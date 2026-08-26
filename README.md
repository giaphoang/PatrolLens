# PatrolLens

PatrolLens is a local-first natural-language search system for body-worn camera footage. It indexes timestamped evidence from transcripts, visible text, frames, temporal windows, and audio features, then optionally asks an OpenRouter video-capable model to verify a small set of candidate clips.

## Quick start

The core package has no mandatory third-party dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Build a dependency-free structural index:

```bash
patrol-lens index ./videos --index .patrol-lens
patrol-lens search "vehicle stopped on the roadside" --index .patrol-lens
```

For real body-camera analysis, install the media extras and enable the desired adapters:

```bash
pip install -e '.[media]'
patrol-lens index ./videos \
  --index .patrol-lens \
  --text-encoder siglip2 \
  --asr faster-whisper \
  --asr-model small.en \
  --visual siglip2 \
  --audio wave \
  --ocr paddle
```

`ffmpeg` and `ffprobe` must be installed separately and available on `PATH`.

## OpenRouter reranking

Set `OPENROUTER_API_KEY` and choose a video-capable model in `OPENROUTER_VLM_MODEL`:

```bash
export OPENROUTER_API_KEY='...'
export OPENROUTER_VLM_MODEL='provider/model-with-video-input'
patrol-lens search \
  "Find moments where a suspect is being handcuffed" \
  --index .patrol-lens \
  --rerank \
  --openrouter-model "$OPENROUTER_VLM_MODEL"
```

Only short candidate clips are sent to the hosted model. Local retrieval remains available if the hosted call fails.

## Data flow

```text
video
  -> ffmpeg metadata/audio/frame extraction
  -> ASR, OCR, visual embeddings, audio features
  -> SQLite FTS + local vector records

query
  -> heuristic modality planner
  -> text / OCR / visual / clip / audio retrieval
  -> reciprocal-rank fusion
  -> optional OpenRouter short-clip reranking
  -> timestamped JSON
```

The default coarse temporal grid uses 16-second windows with an 8-second stride. A 90-minute video produces 674 coarse windows. Frames and audio remain timestamped at their original source positions; full videos are not sent to OpenRouter.

## System design

The formal problem definition, Mermaid architecture, modality mapping, timestamp strategy, and 90-minute video handling are documented in [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md).

## Evaluation format

`patrol-lens evaluate` accepts JSONL records such as:

```json
{"query":"Miranda rights","relevant":[["video-abc",120000,135000]]}
```

The evaluator reports query hit rate and result counts. Add manually labeled intervals for precision, temporal IoU, OCR character accuracy, and reranker ablations.
