"""
DSP layer.

Imports the real processing engine from workspace flute_processor.py and
adds:

- a control-rate compressor (same math as flute_processor.compressor but
  ~1000x faster, required for background jobs on full songs)
- a mastering chain wrapper with progress callbacks
- a stem mixer (per-track gain / pan / mute / solo + master bus)
"""

import numpy as np
import soundfile as sf
from scipy import signal

import flute_processor as fp  # noqa: E402  (workspace root added in app/__init__)

STYLES = fp.STYLES
SUPPORTED_INPUT_FORMATS = fp.SUPPORTED_INPUT_FORMATS

DEFAULT_TARGET_LUFS = fp.DEFAULT_TARGET_LUFS
DEFAULT_STEREO_WIDTH = fp.DEFAULT_STEREO_WIDTH
DEFAULT_PEAK_CEILING_DB = fp.PEAK_CEILING_DB


# ============================================================
# IO helpers
# ============================================================

def load_audio(path):
    """Load any supported file as (samples x 2) float64 + sample rate."""
    return fp.load_audio(str(path))


def save_wav(path, audio, sr):
    """Write 24-bit WAV, sanitizing numerical errors."""
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    sf.write(str(path), audio, sr, subtype="PCM_24")
    return str(path)


def resample_audio(audio, sr, target_sr):
    if sr == target_sr:
        return audio
    out = np.empty(
        (int(round(audio.shape[0] * target_sr / sr)), audio.shape[1]),
        dtype=audio.float() if hasattr(audio, "float") else audio.dtype,
    )
    for ch in range(audio.shape[1]):
        out[:, ch] = signal.resample_poly(audio[:, ch], target_sr, sr)
    return out


# ============================================================
# Fast compressor (control-rate envelope follower)
# ============================================================

def compressor_fast(
    audio,
    sr,
    threshold_db=-22.0,
    ratio=2.5,
    attack_ms=18.0,
    release_ms=130.0,
    control_rate=1000.0,
):
    """
    Same gain law as flute_processor.compressor, but the envelope follower
    runs at ~1 kHz control rate instead of per-sample. For a 4 minute song
    this is ~250k loop iterations instead of ~10M.
    """
    mono = np.mean(audio, axis=1)

    block = max(1, int(round(sr / control_rate)))
    n_blocks = len(mono) // block

    if n_blocks < 4:
        return fp.compressor(
            audio, sr,
            threshold_db=threshold_db, ratio=ratio,
            attack_ms=attack_ms, release_ms=release_ms,
        )

    env_blocks = np.abs(
        mono[: n_blocks * block].reshape(n_blocks, block)
    ).max(axis=1)

    attack_coeff = np.exp(-1.0 / (control_rate * attack_ms / 1000.0))
    release_coeff = np.exp(-1.0 / (control_rate * release_ms / 1000.0))

    env = np.empty(n_blocks, dtype=np.float64)
    prev = 0.0
    for i, value in enumerate(env_blocks):
        coeff = attack_coeff if value > prev else release_coeff
        prev = coeff * prev + (1.0 - coeff) * value
        env[i] = prev

    env_db = fp.linear_to_db(env)

    gain_db = np.zeros_like(env_db)
    over = env_db > threshold_db
    gain_db[over] = (
        threshold_db
        + (env_db[over] - threshold_db) / ratio
        - env_db[over]
    )
    gain = fp.db_to_linear(gain_db)

    t_full = np.arange(len(mono))
    t_ctrl = (np.arange(n_blocks) + 0.5) * block
    gain_full = np.interp(t_full, t_ctrl, gain, left=gain[0], right=gain[-1])

    return audio * gain_full[:, None]


# ============================================================
# Mastering chain (progress-reporting version of process_flute)
# ============================================================

MASTER_STAGES = [
    "Reference EQ",
    "Adaptive EQ",
    "Compression",
    "Saturation",
    "Reverb",
    "Delay",
    "Stereo widening",
    "Safety filters",
    "Limiter",
    "Loudness normalization",
    "Final peak protection",
]


def _noop(*_args, **_kwargs):
    pass


def master_chain(audio, sr, options, on_progress=_noop, on_log=_noop):
    """
    Run the adaptive flute mastering chain on an in-memory buffer.

    options:
        style               one of STYLES
        reference_audio     (audio, sr) tuple or None
        match_amount        0.0 - 1.0
        target_lufs         float
        stereo_width        float
        reverb_percent      None -> style default
        delay_percent       None -> style default
        ceiling_db          peak ceiling
        use_*               per-effect toggles

    Returns (processed_audio, report_dict).
    """
    style = options.get("style", "studio")
    if style not in STYLES:
        raise ValueError(f"Unknown style: {style}")
    settings = STYLES[style]

    match_amount = float(np.clip(options.get("match_amount", 0.0), 0.0, 1.0))
    target_lufs = float(options.get("target_lufs", DEFAULT_TARGET_LUFS))
    stereo_width = float(options.get("stereo_width", DEFAULT_STEREO_WIDTH))
    ceiling_db = float(options.get("ceiling_db", DEFAULT_PEAK_CEILING_DB))

    reverb_percent = options.get("reverb_percent")
    if reverb_percent is None:
        reverb_percent = settings["reverb"]
    delay_percent = options.get("delay_percent")
    if delay_percent is None:
        delay_percent = settings["delay_percent"] if "delay_percent" in settings else settings["delay"]

    reverb_percent = float(np.clip(reverb_percent, 0.0, 100.0))
    delay_percent = float(np.clip(delay_percent, 0.0, 100.0))

    use_eq = options.get("use_eq", True)
    use_compression = options.get("use_compression", True)
    use_saturation = options.get("use_saturation", True)
    use_reverb = options.get("use_reverb", True)
    use_delay = options.get("use_delay", True)
    use_stereo = options.get("use_stereo", True)
    use_limiter = options.get("use_limiter", True)

    total = len(MASTER_STAGES)
    processed = audio.copy()

    on_log(f"Analyzing source ({len(audio) / sr:.1f}s @ {sr} Hz)...")
    source_analysis = fp.analyze_reference_audio(audio, sr)

    reference_analysis = None
    comparison = None
    reference_audio = options.get("reference_audio")

    if reference_audio is not None and match_amount > 0:
        ref_wave, ref_sr = reference_audio
        on_log("Analyzing reference...")
        reference_analysis = fp.analyze_reference_audio(ref_wave, ref_sr)
        comparison = fp.compare_reference(
            source_analysis, reference_analysis, match_amount
        )
        on_log(
            f"Reference: LUFS {reference_analysis['integrated_lufs']:.2f} "
            f"(source {source_analysis['integrated_lufs']:.2f}), "
            f"correction {comparison['loudness_correction']:+.2f} dB"
        )

    def stage(idx, name):
        on_progress(idx / total, name)
        on_log(f"[{idx + 1}/{total}] {name}...")

    # --- Reference tonal matching -------------------------------------
    stage(0, MASTER_STAGES[0])
    if comparison is not None and use_eq and match_amount > 0:
        processed = fp.apply_reference_eq(processed, sr, comparison)

    # --- Adaptive EQ ---------------------------------------------------
    stage(1, MASTER_STAGES[1])
    if use_eq:
        processed = fp.adaptive_eq(
            processed, sr, fp.analyze_audio(processed, sr), style
        )

    # --- Compression ---------------------------------------------------
    stage(2, MASTER_STAGES[2])
    if use_compression:
        if comparison is not None:
            cs = fp.reference_compression_settings(
                source_analysis, reference_analysis, match_amount
            )
            on_log(
                f"Reference compression: {cs['ratio']:.2f}:1 "
                f"@ {cs['threshold']:.1f} dB"
            )
        else:
            cs = {
                "ratio": settings["compression"],
                "threshold": -22.0,
                "attack_ms": 18.0,
                "release_ms": 130.0,
            }
        processed = compressor_fast(
            processed, sr,
            threshold_db=cs["threshold"],
            ratio=cs["ratio"],
            attack_ms=cs["attack_ms"],
            release_ms=cs["release_ms"],
        )

    # --- Saturation ----------------------------------------------------
    stage(3, MASTER_STAGES[3])
    if use_saturation:
        processed = fp.saturation(processed, amount=settings["saturation"])

    # --- Reverb --------------------------------------------------------
    stage(4, MASTER_STAGES[4])
    if use_reverb and reverb_percent > 0:
        processed = fp.apply_reverb(
            processed, sr, percentage=reverb_percent, decay=2.2
        )

    # --- Delay ---------------------------------------------------------
    stage(5, MASTER_STAGES[5])
    if use_delay and delay_percent > 0:
        processed = fp.apply_delay(processed, sr, percentage=delay_percent)

    # --- Stereo widening ------------------------------------------------
    stage(6, MASTER_STAGES[6])
    if use_stereo:
        width = stereo_width
        if comparison is not None:
            width *= comparison["width_ratio"]
        width = float(np.clip(width, 0.70, 1.50))
        processed = fp.stereo_widen(processed, width=width)

    # --- Safety filters --------------------------------------------------
    stage(7, MASTER_STAGES[7])
    processed = fp.butter_filter(processed, sr, 90, "highpass", order=2)
    high_cut = min(19000, sr * 0.45)
    processed = fp.butter_filter(processed, sr, high_cut, "lowpass", order=2)

    # --- Limiter ---------------------------------------------------------
    stage(8, MASTER_STAGES[8])
    if use_limiter:
        processed = fp.limiter(processed, ceiling_db=ceiling_db)

    # --- Loudness --------------------------------------------------------
    final_target_lufs = target_lufs
    if comparison is not None and match_amount > 0:
        final_target_lufs = (
            target_lufs * (1.0 - match_amount)
            + reference_analysis["integrated_lufs"] * match_amount
        )

    stage(9, MASTER_STAGES[9])
    processed, measured_lufs, applied_gain = fp.normalize_loudness(
        processed, sr, final_target_lufs
    )

    # --- Final peak protection -------------------------------------------
    stage(10, MASTER_STAGES[10])
    if use_limiter:
        processed = fp.limiter(processed, ceiling_db=ceiling_db)

    processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)

    on_progress(1.0, "Complete")
    on_log(
        f"Done: target {final_target_lufs:.2f} LUFS, "
        f"gain {applied_gain:+.2f} dB"
    )

    final_analysis = fp.analyze_reference_audio(processed, sr)

    report = {
        "style": style,
        "target_lufs": final_target_lufs,
        "measured_lufs_before": float(measured_lufs),
        "gain_applied_db": float(applied_gain),
        "reverb_percent": reverb_percent,
        "delay_percent": delay_percent,
        "stereo_width": stereo_width,
        "ceiling_db": ceiling_db,
        "effects": {
            "eq": use_eq,
            "compression": use_compression,
            "saturation": use_saturation,
            "reverb": use_reverb and reverb_percent > 0,
            "delay": use_delay and delay_percent > 0,
            "stereo": use_stereo,
            "limiter": use_limiter,
        },
        "before": _analysis_summary(source_analysis),
        "after": _analysis_summary(final_analysis),
        "reference": (
            _analysis_summary(reference_analysis)
            if reference_analysis is not None
            else None
        ),
        "band_corrections": (
            {k: float(v) for k, v in comparison["band_corrections"].items()}
            if comparison is not None
            else None
        ),
    }

    return processed, report


def _analysis_summary(a):
    return {
        "integrated_lufs": float(a["integrated_lufs"]),
        "peak_db": float(a["peak_db"]),
        "rms_db": float(a["rms_db"]),
        "crest_factor_db": float(a["crest_factor_db"]),
        "centroid_hz": float(a["centroid_hz"]),
        "rolloff_hz": float(a["rolloff_hz"]),
        "stereo_width": float(a["stereo_width"]),
        "dynamics_range": float(a["dynamics_range"]),
    }


# ============================================================
# Mixer
# ============================================================

def pan_gains(pan):
    """Constant-power pan. pan in [-1, 1]; center keeps unity gain."""
    pan = float(np.clip(pan, -1.0, 1.0))
    theta = (pan + 1.0) * np.pi / 4.0
    return np.sqrt(2.0) * np.cos(theta), np.sqrt(2.0) * np.sin(theta)


def render_mix(tracks, master=None, on_progress=_noop, on_log=_noop):
    """
    Render a mix from loaded tracks.

    tracks: list of dicts {
        "audio": (samples x 2) float array,
        "sr": int,
        "name": str,
        "gain_db": float,
        "pan": float (-1..1),
        "mute": bool,
        "solo": bool,
    }
    master: {
        "gain_db": float,
        "target_lufs": float or None,
        "ceiling_db": float,
        "limiter": bool,
    }

    Returns (mix_audio, sr, report).
    """
    master = master or {}
    if not tracks:
        raise ValueError("No tracks to mix")

    any_active = any(t.get("solo") for t in tracks)
    active = [
        t for t in tracks
        if (t.get("solo") if any_active else not t.get("mute"))
    ]
    if not active:
        raise ValueError("All tracks are muted or excluded by solo")

    on_progress(0.05, "Loading tracks")
    target_sr = max(t["sr"] for t in active)

    total_len = max(
        int(round(len(t["audio"]) * target_sr / t["sr"])) for t in active
    )
    mix = np.zeros((total_len, 2), dtype=np.float64)

    for i, track in enumerate(active):
        wave = track["audio"]
        if track["sr"] != target_sr:
            wave = resample_audio(wave, track["sr"], target_sr)

        gain = fp.db_to_linear(float(track.get("gain_db", 0.0)))
        gl, gr = pan_gains(track.get("pan", 0.0))

        n = min(len(wave), total_len)
        mix[:n, 0] += wave[:n, 0] * gain * gl
        mix[:n, 1] += wave[:n, 1] * gain * gr

        on_log(
            f"+ {track.get('name', f'track {i + 1}')}: "
            f"{track.get('gain_db', 0.0):+.1f} dB, "
            f"pan {track.get('pan', 0.0):+.2f}"
        )
        on_progress(0.05 + 0.6 * (i + 1) / len(active), "Summing tracks")

    # Master bus
    master_gain_db = float(master.get("gain_db", 0.0))
    if abs(master_gain_db) > 1e-6:
        mix *= fp.db_to_linear(master_gain_db)

    ceiling_db = float(master.get("ceiling_db", DEFAULT_PEAK_CEILING_DB))

    target_lufs = master.get("target_lufs")
    gain_applied = 0.0
    if target_lufs is not None:
        on_progress(0.75, "Normalizing mix loudness")
        mix, measured_lufs, gain_applied = fp.normalize_loudness(
            mix, target_sr, float(target_lufs)
        )
        on_log(
            f"Mix loudness: {measured_lufs:.2f} LUFS "
            f"(gain {gain_applied:+.2f} dB)"
        )

    if master.get("limiter", True):
        on_progress(0.9, "Limiting mix")
        mix = fp.limiter(mix, ceiling_db=ceiling_db)

    mix = np.nan_to_num(mix, nan=0.0, posinf=0.0, neginf=0.0)
    on_progress(1.0, "Complete")

    analysis = fp.analyze_reference_audio(mix, target_sr)
    report = {
        "sample_rate": target_sr,
        "duration": len(mix) / target_sr,
        "tracks_used": len(active),
        "master_gain_db": master_gain_db,
        "target_lufs": float(target_lufs) if target_lufs is not None else None,
        "gain_applied_db": float(gain_applied),
        "ceiling_db": ceiling_db,
        "after": _analysis_summary(analysis),
    }

    return mix, target_sr, report
