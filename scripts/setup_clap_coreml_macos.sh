#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "error: larger_clap_general CoreML setup requires Apple Silicon macOS" >&2
  exit 1
fi

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
model_root="${PATROLLENS_CLAP_MODEL_ROOT:-$repo_root/.patrol-lens-models/larger_clap_general_coreml}"

cd "$repo_root"
uv sync --extra dev --extra full --extra clap-macos

PATROLLENS_CLAP_MODEL_ROOT="$model_root" .venv/bin/python - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

root = Path(os.environ["PATROLLENS_CLAP_MODEL_ROOT"]).expanduser().resolve()
model_dir = root / "model"
tokenizer_dir = root / "tokenizer"

snapshot_download(
    repo_id="atan2f/larger_clap_general_coreml",
    local_dir=model_dir,
    allow_patterns=[
        "clap_audio_encoder.mlpackage/**",
        "text_model.onnx",
        "text_model.onnx.data",
    ],
)
snapshot_download(
    repo_id="Xenova/larger_clap_general",
    local_dir=tokenizer_dir,
    allow_patterns=[
        "added_tokens.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ],
)

print(f"CLAP audio model: {model_dir / 'clap_audio_encoder.mlpackage'}")
print(f"CLAP text model:  {model_dir / 'text_model.onnx'}")
print(f"CLAP tokenizer:   {tokenizer_dir}")
PY

echo "CLAP CoreML setup complete. The full ingestion profile will auto-detect these artifacts."
