"""
Audio library: uploads registry, file analysis and waveform peaks.

Every uploaded / rendered file is registered so the UI can list, play,
analyze and reuse it across separation / mixing / mastering.
"""

import json
import re
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from app import dsp
from app.config import ALLOWED_EXTENSIONS, DATA_DIR, UPLOADS_DIR

REGISTRY_PATH = DATA_DIR / "library.json"

_registry = {}
_registry_lock = threading.Lock()
_analysis_cache = {}


# ============================================================
# Naming / registry
# ============================================================

def _safe_stem(filename):
    stem = Path(filename).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return stem[:80] or "audio"


def _load_registry():
    global _registry
    if REGISTRY_PATH.exists():
        try:
            _registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            _registry = {}
    else:
        _registry = {}


def _save_registry():
    tmp = REGISTRY_PATH.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(_registry, indent=2, sort_keys=True), encoding="utf-8"
    )
    tmp.replace(REGISTRY_PATH)


def validate_filename(filename):
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format '{ext}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    return ext


def save_upload(filename, data):
    """Persist raw upload bytes, register it, return the record."""
    ext = validate_filename(filename)
    file_id = uuid.uuid4().hex[:12]
    stored_name = f"{_safe_stem(filename)}_{file_id}{ext}"
    path = UPLOADS_DIR / stored_name
    path.write_bytes(data)

    record = {
        "id": file_id,
        "kind": "upload",
        "name": filename,
        "path": str(path),
        "url": f"/media/uploads/{stored_name}",
        "size_bytes": len(data),
        "created_at": time.time(),
    }

    with _registry_lock:
        _registry[file_id] = record
        _save_registry()

    return record


def register_file(path, kind, name=None, extra=None):
    """Register an already-written file (rendered mix / master / stem)."""
    path = Path(path)
    file_id = uuid.uuid4().hex[:12]
    record = {
        "id": file_id,
        "kind": kind,
        "name": name or path.name,
        "path": str(path),
        "url": _media_url(path),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "created_at": time.time(),
    }
    if extra:
        record.update(extra)

    with _registry_lock:
        _registry[file_id] = record
        _save_registry()

    return record


def _media_url(path):
    path = Path(path).resolve()
    try:
        rel = path.relative_to(DATA_DIR.resolve())
        return "/media/" + rel.as_posix()
    except ValueError:
        return "/media/uploads/" + path.name


def get_file(file_id):
    with _registry_lock:
        record = _registry.get(file_id)
    if record is None or not Path(record["path"]).exists():
        return None
    return record


def list_files(kind=None):
    with _registry_lock:
        records = list(_registry.values())
    records = [r for r in records if Path(r["path"]).exists()]
    if kind:
        records = [r for r in records if r.get("kind") == kind]
    records.sort(key=lambda r: r.get("created_at", 0), reverse=True)
    return records


def delete_file(file_id):
    with _registry_lock:
        record = _registry.pop(file_id, None)
        if record is not None:
            _save_registry()
    if record is None:
        return False

    path = Path(record["path"])
    _analysis_cache.pop(str(path), None)

    # On Windows the file may still be open (e.g. streaming to the
    # browser). Retry briefly; if it stays locked, leave it on disk —
    # the registry entry is gone, so it simply disappears from the UI.
    for attempt in range(4):
        try:
            if path.exists():
                path.unlink()
            break
        except OSError:
            if attempt == 3:
                break
            time.sleep(0.25 * (attempt + 1))

    return True


def register_stem_set(manifest, name=None):
    """Register a separation result (one record per stem + a set record)."""
    stems = []
    for stem in manifest["stems"]:
        rec = register_file(
            stem["file"],
            kind="stem",
            name=f"{name or Path(manifest['source']).stem} — {stem['name']}",
            extra={"stem": stem["name"], "set": manifest["job_id"]},
        )
        stems.append({**stem, "id": rec["id"], "url": rec["url"]})

    set_record = register_file(
        manifest["stems"][0]["file"],
        kind="stemset",
        name=name or Path(manifest["source"]).stem,
        extra={
            "stems": stems,
            "model": manifest["model"],
            "duration": manifest["duration"],
            "sample_rate": manifest["sample_rate"],
            "source": manifest["source"],
        },
    )
    return set_record


# ============================================================
# Analysis
# ============================================================

def analyze_file(path, buckets=1600):
    """Full analysis: loudness/spectrum summary + waveform peaks."""
    key = str(path)
    cached = _analysis_cache.get(key)
    if cached is not None:
        return cached

    audio, sr = dsp.load_audio(path)
    summary = dsp.fp.analyze_reference_audio(audio, sr)

    mono = np.mean(audio, axis=1)
    duration = len(mono) / sr

    # Waveform min/max peaks, decimated to `buckets` columns
    n = min(buckets, len(mono))
    chunk = len(mono) // n or 1
    trimmed = mono[: n * chunk].reshape(n, chunk)
    peaks_min = trimmed.min(axis=1)
    peaks_max = trimmed.max(axis=1)

    # Average magnitude spectrum on a log-ish grid
    fft = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    freqs = np.fft.rfftfreq(len(mono), 1.0 / sr)
    grid = np.geomspace(30, min(20000, sr / 2), 240)
    spectrum = np.interp(grid, freqs, fft)
    spectrum_db = 20 * np.log10(np.maximum(spectrum, 1e-9))
    spectrum_db -= spectrum_db.max()

    result = {
        "sample_rate": sr,
        "channels": int(audio.shape[1]),
        "duration": round(duration, 3),
        "integrated_lufs": round(float(summary["integrated_lufs"]), 2),
        "peak_db": round(float(summary["peak_db"]), 2),
        "rms_db": round(float(summary["rms_db"]), 2),
        "crest_factor_db": round(float(summary["crest_factor_db"]), 2),
        "centroid_hz": round(float(summary["centroid_hz"]), 1),
        "rolloff_hz": round(float(summary["rolloff_hz"]), 1),
        "stereo_width": round(float(summary["stereo_width"]), 3),
        "dynamics_range": round(float(summary["dynamics_range"]), 2),
        "waveform": {
            "min": [round(float(v), 4) for v in peaks_min],
            "max": [round(float(v), 4) for v in peaks_max],
        },
        "spectrum": {
            "freqs": [round(float(v), 1) for v in grid],
            "db": [round(float(v), 2) for v in spectrum_db],
        },
    }

    _analysis_cache[key] = result
    return result


_load_registry()
