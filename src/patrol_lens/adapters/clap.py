from __future__ import annotations

import math
import shutil
import subprocess
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, Protocol


DEFAULT_CLAP_MODEL = "atan2f/larger_clap_general_coreml"
CLAP_DIMENSIONS = 512
CLAP_SAMPLE_RATE = 48_000
CLAP_WINDOW_MS = 10_000
CLAP_WINDOW_SAMPLES = CLAP_SAMPLE_RATE * CLAP_WINDOW_MS // 1000


class ClapEmbeddingDimensionError(RuntimeError):
    def __init__(self, actual: int) -> None:
        super().__init__(f"Expected {CLAP_DIMENSIONS} CLAP dimensions, got {actual}")


class AudioEmbeddingBackend(Protocol):
    model_name: str
    dimensions: int

    def encode_audio_windows(
        self,
        media_path: str | Path,
        intervals: Sequence[tuple[int, int]],
    ) -> Iterable[list[float]]: ...

    def encode_text(self, text: str) -> list[float]: ...


def clap_intervals(
    duration_ms: int,
    *,
    window_ms: int = CLAP_WINDOW_MS,
    stride_ms: int = 5_000,
) -> list[tuple[int, int]]:
    """Return fixed-window CLAP timestamps, with only the final window padded."""

    if duration_ms <= 0:
        return []
    if window_ms != CLAP_WINDOW_MS:
        raise ValueError("larger_clap_general CoreML requires a fixed 10-second window")
    if stride_ms <= 0 or stride_ms > window_ms:
        raise ValueError("CLAP stride must be between 1 ms and 10 seconds")
    intervals: list[tuple[int, int]] = []
    start_ms = 0
    while start_ms < duration_ms:
        end_ms = min(duration_ms, start_ms + window_ms)
        intervals.append((start_ms, end_ms))
        if end_ms >= duration_ms:
            break
        start_ms += stride_ms
    return intervals


def _normalized_vector(values: Any) -> list[float]:
    if hasattr(values, "reshape"):
        values = values.reshape(-1).tolist()
    vector = [float(value) for value in values]
    if len(vector) != CLAP_DIMENSIONS:
        raise ClapEmbeddingDimensionError(len(vector))
    if not all(math.isfinite(value) for value in vector):
        raise RuntimeError("CLAP embedding contains non-finite values")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-12:
        raise RuntimeError("CLAP embedding has zero norm")
    return [value / norm for value in vector]


def _read_exact(stream: Any, byte_count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = byte_count
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class ClapCoreMLBackend:
    """Paired CoreML-audio/ONNX-text larger_clap_general encoders.

    Audio is decoded directly from the original media as streamed 48 kHz mono
    float32 PCM. No second full-length WAV is written to disk.
    """

    dimensions = CLAP_DIMENSIONS
    sample_rate = CLAP_SAMPLE_RATE
    window_ms = CLAP_WINDOW_MS

    def __init__(
        self,
        audio_model_path: str | Path,
        text_model_path: str | Path,
        tokenizer_path: str | Path,
        *,
        model_name: str = DEFAULT_CLAP_MODEL,
        ffmpeg: str = "ffmpeg",
        compute_units: str = "cpu_only",
    ) -> None:
        self.audio_model_path = Path(audio_model_path).expanduser().resolve()
        self.text_model_path = Path(text_model_path).expanduser().resolve()
        self.tokenizer_path = Path(tokenizer_path).expanduser().resolve()
        self.model_name = model_name
        self.ffmpeg = ffmpeg
        self.compute_units = compute_units
        self._audio_model: Any | None = None
        self._text_session: Any | None = None
        self._tokenizer: Any | None = None

    def validate_setup(self, *, require_text: bool = True) -> None:
        missing = [
            str(path)
            for path in (self.audio_model_path, self.text_model_path, self.tokenizer_path)
            if not path.exists() and (require_text or path == self.audio_model_path)
        ]
        if missing:
            raise RuntimeError(
                "CLAP model artifacts are missing: "
                + ", ".join(missing)
                + "; run scripts/setup_clap_coreml_macos.sh"
            )
        if shutil.which(self.ffmpeg) is None:
            raise RuntimeError("ffmpeg is required for CLAP audio decoding")

    def _load_audio_model(self) -> Any:
        if self._audio_model is None:
            self.validate_setup(require_text=False)
            try:
                import coremltools as ct
            except ImportError as exc:
                raise RuntimeError(
                    "coremltools is not installed; install patrol-lens[clap-macos]"
                ) from exc
            units = {
                "cpu_only": ct.ComputeUnit.CPU_ONLY,
                "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
                "all": ct.ComputeUnit.ALL,
            }.get(self.compute_units)
            if units is None:
                raise ValueError(
                    "CLAP compute units must be cpu_only, cpu_and_gpu, or all"
                )
            self._audio_model = ct.models.MLModel(
                str(self.audio_model_path),
                compute_units=units,
            )
        return self._audio_model

    def _load_text(self) -> tuple[Any, Any]:
        if self._text_session is None or self._tokenizer is None:
            self.validate_setup()
            try:
                import onnxruntime as ort
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "onnxruntime and transformers are required; install "
                    "patrol-lens[clap-macos]"
                ) from exc
            self._text_session = ort.InferenceSession(
                str(self.text_model_path),
                providers=["CPUExecutionProvider"],
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.tokenizer_path),
                local_files_only=True,
            )
        return self._text_session, self._tokenizer

    def encode_text(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("CLAP text query must not be empty")
        session, tokenizer = self._load_text()
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy is required for CLAP") from exc
        encoded = tokenizer(text, return_tensors="np", padding=True)
        outputs = session.run(
            ["text_embeds"],
            {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
            },
        )
        return _normalized_vector(outputs[0])

    def _predict_waveform(self, waveform: Any) -> list[float]:
        output = self._load_audio_model().predict({"audio": waveform})
        if "embedding" not in output:
            raise RuntimeError("CLAP CoreML output does not contain 'embedding'")
        return _normalized_vector(output["embedding"])

    def encode_audio_windows(
        self,
        media_path: str | Path,
        intervals: Sequence[tuple[int, int]],
    ) -> Iterator[list[float]]:
        """Decode once and yield one 512-d vector per requested interval."""

        if not intervals:
            return
        ordered = list(intervals)
        if ordered != sorted(ordered) or any(end <= start for start, end in ordered):
            raise ValueError("CLAP intervals must be ordered, non-empty timestamps")
        if any(end - start > self.window_ms for start, end in ordered):
            raise ValueError("CLAP intervals cannot exceed 10 seconds")
        self.validate_setup(require_text=False)
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy is required for CLAP") from exc

        command = [
            self.ffmpeg,
            "-v",
            "error",
            "-i",
            str(Path(media_path).expanduser()),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(self.sample_rate),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-",
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdout is not None
        assert process.stderr is not None
        buffer = np.empty(0, dtype=np.float32)
        buffer_start = 0
        decoded_end = 0
        eof = False
        completed = False
        try:
            for start_ms, _end_ms in ordered:
                start_sample = round(start_ms * self.sample_rate / 1000)
                target_end = start_sample + CLAP_WINDOW_SAMPLES

                # Retain only overlap needed by the next requested window. If
                # cache hits leave a large timestamp gap, discard decoded PCM
                # directly instead of allowing a full-file array to grow.
                buffered_drop = min(
                    max(0, start_sample - buffer_start),
                    int(buffer.size),
                )
                if buffered_drop:
                    buffer = buffer[buffered_drop:]
                    buffer_start += buffered_drop
                while decoded_end < start_sample and not eof:
                    requested = min(262_144, start_sample - decoded_end)
                    raw = _read_exact(process.stdout, requested * 4)
                    if not raw:
                        eof = True
                        break
                    samples_read = len(raw) // 4
                    decoded_end += samples_read
                    buffer_start = decoded_end
                    if samples_read < requested:
                        eof = True

                while decoded_end < target_end and not eof:
                    requested = min(262_144, target_end - decoded_end)
                    raw = _read_exact(process.stdout, requested * 4)
                    if not raw:
                        eof = True
                        break
                    usable = len(raw) - (len(raw) % 4)
                    samples = np.frombuffer(raw[:usable], dtype="<f4").copy()
                    if samples.size == 0:
                        eof = True
                        break
                    buffer = np.concatenate((buffer, samples))
                    decoded_end += int(samples.size)
                    if samples.size < requested:
                        eof = True

                if eof:
                    return_code = process.wait()
                    if return_code:
                        stderr = process.stderr.read().decode(
                            "utf-8",
                            errors="replace",
                        ).strip()
                        raise RuntimeError(
                            f"CLAP audio decoding failed: {stderr or 'ffmpeg failed'}"
                        )

                offset = max(0, start_sample - buffer_start)
                waveform = buffer[offset : offset + CLAP_WINDOW_SAMPLES]
                if waveform.size < CLAP_WINDOW_SAMPLES:
                    waveform = np.pad(
                        waveform,
                        (0, CLAP_WINDOW_SAMPLES - waveform.size),
                    )
                peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
                if peak > 1e-8:
                    waveform = waveform / peak
                model_input = np.asarray(waveform, dtype=np.float32).reshape(
                    1, CLAP_WINDOW_SAMPLES
                )

                yield self._predict_waveform(model_input)
            completed = True
        finally:
            process.stdout.close()
            was_running = process.poll() is None
            if was_running:
                process.terminate()
            stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
            process.stderr.close()
            return_code = process.wait()
            if completed and not was_running and return_code:
                raise RuntimeError(f"CLAP audio decoding failed: {stderr or 'ffmpeg failed'}")
