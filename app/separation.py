"""
Stem separation via Demucs (htdemucs_6s by default).

Uses the demucs Python API so the UI gets real progress updates from the
model callback instead of parsing subprocess output. The heavy model is
loaded once and reused; a lock serializes GPU/CPU work.
"""

import threading
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from app.config import STEMS_DIR

# Model -> ordered stem list
MODELS = {
    "htdemucs_6s": {
        "label": "htdemucs (6 stems: drums, bass, other, vocals, guitar, piano)",
        "sources": ["drums", "bass", "other", "vocals", "guitar", "piano"],
    },
    "htdemucs": {
        "label": "htdemucs (4 stems: drums, bass, other, vocals)",
        "sources": ["drums", "bass", "other", "vocals"],
    },
}

DEFAULT_MODEL = "htdemucs_6s"

_separators = {}
_separator_lock = threading.Lock()
_inference_lock = threading.Lock()


def device_name():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_separator(model_name):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    with _separator_lock:
        sep = _separators.get(model_name)
        if sep is None:
            from demucs.api import Separator

            sep = Separator(model=model_name, device=device_name())
            _separators[model_name] = sep
        return sep


def separate_file(input_path, job, model_name=DEFAULT_MODEL):
    """
    Separate a file into stems. Writes 24-bit WAVs to
    data/stems/<job_id>/ and returns a manifest dict.

    `job` needs: set_progress(frac, stage), log(msg), raise_if_cancelled().
    """
    from demucs.api import save_audio

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    separator = get_separator(model_name)
    sr = separator.samplerate
    out_dir = STEMS_DIR / job.id
    out_dir.mkdir(parents=True, exist_ok=True)

    job.log(f"Loading audio: {input_path.name}")
    job.set_progress(0.02, "Loading audio")

    wav = separator._load_audio(input_path)  # (channels, samples) float32
    duration = wav.shape[1] / sr
    job.log(
        f"Track duration: {duration:.1f}s @ {sr} Hz, "
        f"device: {device_name()}"
    )

    audio_length = wav.shape[1]
    state = {"last": 0.0}

    def callback(info):
        job.raise_if_cancelled()
        offset = info.get("segment_offset", 0)
        length = info.get("audio_length") or audio_length
        fraction = min(0.98, 0.05 + 0.93 * (offset / max(1, length)))
        if fraction > state["last"]:
            state["last"] = fraction
            job.set_progress(fraction, "Separating stems")

    separator.update_parameter(callback=callback)

    with _inference_lock:
        job.set_progress(0.05, "Separating stems")
        job.log(f"Running Demucs ({model_name})...")
        _origin, stems = separator.separate_tensor(wav, sr)

    job.raise_if_cancelled()
    job.set_progress(0.98, "Saving stems")

    manifest_stems = []
    for name in MODELS[model_name]["sources"]:
        if name not in stems:
            continue
        stem = stems[name]
        if stem.ndim == 1:
            stem = np.stack([stem, stem])
        file_path = out_dir / f"{name}.wav"
        save_audio(stem.cpu(), file_path, samplerate=sr, bits_per_sample=24)
        manifest_stems.append(
            {
                "name": name,
                "file": str(file_path),
                "url": f"/media/stems/{job.id}/{name}.wav",
            }
        )
        job.log(f"Saved stem: {name}")

    manifest = {
        "job_id": job.id,
        "source": str(input_path),
        "model": model_name,
        "sample_rate": sr,
        "duration": duration,
        "stems": manifest_stems,
    }

    job.set_progress(1.0, "Complete")
    return manifest
