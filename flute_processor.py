
"""
ADAPTIVE FLUTE STUDIO PROCESSOR + REFERENCE MATCHING
====================================================

Supported input:
WAV / MP3 / M4A / FLAC / AAC / OGG / OPUS

Output:
24-bit WAV

Features:
- Automatic flute analysis
- Adaptive EQ
- Sweet/clear flute profiles
- High-register flute profile
- Compression
- Saturation
- Reverb with percentage control
- Delay with percentage control
- Stereo widening
- Limiter
- LUFS normalization
- Reference audio analysis
- Reference tonal matching
- Reference dynamics matching
- Reference stereo matching
- Reference loudness matching
- Detailed reference report

Examples
--------

Normal studio:

    python flute_processor.py flute.m4a

Sweet and clear:

    python flute_processor.py flute.m4a \
        --style sweet_clear

Sweet, clear and high-register:

    python flute_processor.py flute.m4a \
        --style sweet_clear_high

Sweet clear with controlled ambience:

    python flute_processor.py flute.m4a \
        --style sweet_clear \
        --reverb 15 \
        --delay 8

High flute with ambience:

    python flute_processor.py flute.m4a \
        --style sweet_clear_high \
        --reverb 12 \
        --delay 6

Reference matching:

    python flute_processor.py flute.m4a \
        --reference reference.m4a \
        --match 80

Reference matching with sweet flute:

    python flute_processor.py flute.m4a \
        --reference reference.m4a \
        --match 80 \
        --style sweet_clear \
        --reverb 12 \
        --delay 6

Disable reverb:

    python flute_processor.py flute.m4a \
        --style sweet_clear \
        --reverb 0

Disable delay:

    python flute_processor.py flute.m4a \
        --style sweet_clear \
        --delay 0

Disable multiple effects:

    python flute_processor.py flute.m4a \
        --style sweet_clear \
        --reverb 0 \
        --delay 0 \
        --saturation off
"""

from pathlib import Path
import argparse

import numpy as np
import librosa
import soundfile as sf

from scipy import signal
import pyloudnorm as pyln


# ============================================================
# DEFAULT SETTINGS
# ============================================================

DEFAULT_INPUT = "flute.m4a"
DEFAULT_OUTPUT = "flute_processed.wav"

DEFAULT_STYLE = "studio"

DEFAULT_TARGET_LUFS = -14.0

PEAK_CEILING_DB = -1.0

DEFAULT_STEREO_WIDTH = 1.12

DEFAULT_REVERB_PERCENT = 15.0
DEFAULT_DELAY_PERCENT = 5.0


# ============================================================
# SUPPORTED FILE FORMATS
# ============================================================

SUPPORTED_INPUT_FORMATS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".aac",
    ".ogg",
    ".opus",
}


# ============================================================
# SOUND STYLES
# ============================================================

STYLES = {

    # --------------------------------------------------------
    # Original profiles
    # --------------------------------------------------------

    "warm": {
        "body": 1.5,
        "presence": -1.0,
        "brightness": -1.0,
        "air": -1.0,
        "reverb": 22.0,
        "delay": 4.0,
        "compression": 2.5,
        "saturation": 0.10,
        "harsh_cut": -1.5,
        "air_cut": 0.0,
    },

    "natural": {
        "body": 0.5,
        "presence": 0.0,
        "brightness": 0.0,
        "air": 0.5,
        "reverb": 15.0,
        "delay": 3.0,
        "compression": 2.0,
        "saturation": 0.05,
        "harsh_cut": -1.0,
        "air_cut": 0.0,
    },

    "bright": {
        "body": -0.5,
        "presence": 2.0,
        "brightness": 2.0,
        "air": 3.0,
        "reverb": 14.0,
        "delay": 3.0,
        "compression": 2.5,
        "saturation": 0.06,
        "harsh_cut": -1.0,
        "air_cut": 0.0,
    },

    "cinematic": {
        "body": 1.0,
        "presence": 1.0,
        "brightness": 1.0,
        "air": 2.0,
        "reverb": 30.0,
        "delay": 8.0,
        "compression": 2.0,
        "saturation": 0.10,
        "harsh_cut": -1.5,
        "air_cut": 0.0,
    },

    "studio": {
        "body": 0.5,
        "presence": 1.0,
        "brightness": 0.5,
        "air": 1.5,
        "reverb": 18.0,
        "delay": 4.0,
        "compression": 2.5,
        "saturation": 0.08,
        "harsh_cut": -1.5,
        "air_cut": 0.0,
    },

    # --------------------------------------------------------
    # NEW: Sweet + Clear
    #
    # Designed for:
    # - sweet flute tone
    # - clear articulation
    # - less breathiness
    # - less harshness
    # - polished studio sound
    # --------------------------------------------------------

    "sweet_clear": {
        "body": 0.8,
        "presence": 0.5,
        "brightness": -0.5,
        "air": 0.3,
        "reverb": 15.0,
        "delay": 4.0,
        "compression": 2.2,
        "saturation": 0.025,
        "harsh_cut": -2.2,
        "air_cut": -0.5,
    },

    # --------------------------------------------------------
    # NEW: Sweet + Clear High Register
    #
    # Designed specifically for:
    # - high-register flute
    # - high C / high notes
    # - bright flute recordings
    # - avoiding painful 3-8 kHz harshness
    # - avoiding distortion
    #
    # This profile intentionally uses:
    # - less saturation
    # - gentler high-frequency boost
    # - stronger harshness control
    # - slightly softer compression
    # - controlled air
    # --------------------------------------------------------

    "sweet_clear_high": {
        "body": 0.6,
        "presence": -0.5,
        "brightness": -1.0,
        "air": -0.5,
        "reverb": 12.0,
        "delay": 4.0,
        "compression": 1.8,
        "saturation": 0.008,
        "harsh_cut": -3.0,
        "air_cut": -1.0,
    },
}


# ============================================================
# DB UTILITIES
# ============================================================

def db_to_linear(db):
    return 10.0 ** (db / 20.0)


def linear_to_db(value):
    value = np.maximum(
        np.abs(value),
        1e-12
    )

    return 20.0 * np.log10(value)


# ============================================================
# FILE VALIDATION
# ============================================================

def validate_input_file(path):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"\nAudio file not found:\n{path}"
        )

    if not path.is_file():
        raise ValueError(
            f"\nNot a file:\n{path}"
        )

    extension = path.suffix.lower()

    if extension not in SUPPORTED_INPUT_FORMATS:

        supported = ", ".join(
            sorted(SUPPORTED_INPUT_FORMATS)
        )

        raise ValueError(
            f"\nUnsupported format: {extension}\n"
            f"Supported formats: {supported}"
        )

    return path


# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio(input_file):

    input_file = validate_input_file(
        input_file
    )

    print(
        f"Loading: {input_file}"
    )

    print(
        f"Format: {input_file.suffix.lower()}"
    )

    try:

        audio, sr = librosa.load(
            str(input_file),
            sr=None,
            mono=False
        )

    except Exception as exc:

        if input_file.suffix.lower() in {
            ".m4a",
            ".aac"
        }:

            raise RuntimeError(
                "\nCould not decode the M4A/AAC file.\n\n"
                "Make sure FFmpeg is installed and "
                "available in your system PATH.\n\n"
                f"Original error:\n{exc}"
            ) from exc

        raise RuntimeError(
            f"\nCould not decode:\n"
            f"{input_file}\n\n"
            f"Original error:\n{exc}"
        ) from exc

    # --------------------------------------------------------
    # Convert to samples x channels
    # --------------------------------------------------------

    if audio.ndim == 1:

        audio = np.column_stack([
            audio,
            audio
        ])

    else:

        audio = audio.T

        if audio.shape[1] == 1:

            audio = np.column_stack([
                audio[:, 0],
                audio[:, 0]
            ])

        elif audio.shape[1] > 2:

            audio = audio[:, :2]

    return audio.astype(
        np.float64,
        copy=False
    ), sr


# ============================================================
# BASIC AUDIO ANALYSIS
# ============================================================

def analyze_audio(audio, sr):

    mono = np.mean(
        audio,
        axis=1
    )

    rms = np.sqrt(
        np.mean(mono ** 2)
    )

    peak = np.max(
        np.abs(mono)
    )

    rms_db = linear_to_db(rms)
    peak_db = linear_to_db(peak)

    centroid = librosa.feature.spectral_centroid(
        y=mono,
        sr=sr
    )

    bandwidth = librosa.feature.spectral_bandwidth(
        y=mono,
        sr=sr
    )

    rolloff = librosa.feature.spectral_rolloff(
        y=mono,
        sr=sr,
        roll_percent=0.85
    )

    centroid_hz = float(
        np.median(centroid)
    )

    bandwidth_hz = float(
        np.median(bandwidth)
    )

    rolloff_hz = float(
        np.median(rolloff)
    )

    flatness = librosa.feature.spectral_flatness(
        y=mono
    )

    flatness_value = float(
        np.median(flatness)
    )

    median_f0 = 0.0

    try:

        f0, voiced_flag, voiced_prob = librosa.pyin(
            mono,
            fmin=librosa.note_to_hz("C3"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr
        )

        valid_f0 = f0[
            np.isfinite(f0)
        ]

        if len(valid_f0) > 0:

            median_f0 = float(
                np.median(valid_f0)
            )

    except Exception:

        median_f0 = 0.0

    return {
        "rms_db": rms_db,
        "peak_db": peak_db,
        "centroid_hz": centroid_hz,
        "bandwidth_hz": bandwidth_hz,
        "rolloff_hz": rolloff_hz,
        "flatness": flatness_value,
        "median_f0": median_f0,
    }


# ============================================================
# DETAILED REFERENCE ANALYSIS
# ============================================================

def analyze_reference_audio(audio, sr):

    mono = np.mean(
        audio,
        axis=1
    )

    # --------------------------------------------------------
    # Levels
    # --------------------------------------------------------

    rms = np.sqrt(
        np.mean(mono ** 2)
    )

    peak = np.max(
        np.abs(mono)
    )

    rms_db = linear_to_db(rms)
    peak_db = linear_to_db(peak)

    crest_factor_db = (
        peak_db - rms_db
    )

    # --------------------------------------------------------
    # Spectrum
    # --------------------------------------------------------

    centroid = librosa.feature.spectral_centroid(
        y=mono,
        sr=sr
    )

    rolloff = librosa.feature.spectral_rolloff(
        y=mono,
        sr=sr,
        roll_percent=0.85
    )

    bandwidth = librosa.feature.spectral_bandwidth(
        y=mono,
        sr=sr
    )

    centroid_hz = float(
        np.median(centroid)
    )

    rolloff_hz = float(
        np.median(rolloff)
    )

    bandwidth_hz = float(
        np.median(bandwidth)
    )

    # --------------------------------------------------------
    # Frequency bands
    # --------------------------------------------------------

    frequencies, power = signal.periodogram(
        mono,
        fs=sr
    )

    def band_db(low, high):

        high = min(
            high,
            sr / 2.0
        )

        if high <= low:
            return -100.0

        mask = (
            (frequencies >= low)
            &
            (frequencies < high)
        )

        if not np.any(mask):
            return -100.0

        energy = np.mean(
            power[mask]
        )

        return linear_to_db(
            np.sqrt(
                max(
                    energy,
                    1e-20
                )
            )
        )

    bands = {

        "sub": band_db(
            20,
            80
        ),

        "low": band_db(
            80,
            250
        ),

        "low_mid": band_db(
            250,
            500
        ),

        "mid": band_db(
            500,
            1000
        ),

        "upper_mid": band_db(
            1000,
            2000
        ),

        "presence": band_db(
            2000,
            5000
        ),

        "brightness": band_db(
            5000,
            8000
        ),

        "air": band_db(
            8000,
            min(
                16000,
                sr / 2
            )
        ),
    }

    # --------------------------------------------------------
    # Stereo analysis
    # --------------------------------------------------------

    if audio.shape[1] >= 2:

        left = audio[:, 0]
        right = audio[:, 1]

        mid = (
            left + right
        ) * 0.5

        side = (
            left - right
        ) * 0.5

        mid_rms = np.sqrt(
            np.mean(mid ** 2)
        )

        side_rms = np.sqrt(
            np.mean(side ** 2)
        )

        stereo_width_db = (
            linear_to_db(side_rms)
            -
            linear_to_db(mid_rms)
        )

        stereo_width = (
            side_rms
            /
            max(
                mid_rms,
                1e-12
            )
        )

    else:

        stereo_width_db = -100.0
        stereo_width = 0.0

    # --------------------------------------------------------
    # LUFS
    # --------------------------------------------------------

    meter = pyln.Meter(sr)

    try:

        integrated_lufs = (
            meter.integrated_loudness(
                mono
            )
        )

        if not np.isfinite(
            integrated_lufs
        ):

            integrated_lufs = (
                rms_db - 0.691
            )

    except Exception:

        integrated_lufs = (
            rms_db - 0.691
        )

    # --------------------------------------------------------
    # Dynamics
    # --------------------------------------------------------

    frame_length = int(
        sr * 3.0
    )

    hop_length = int(
        sr * 1.0
    )

    if len(mono) >= frame_length:

        frame_rms = librosa.feature.rms(
            y=mono,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]

        frame_db = linear_to_db(
            frame_rms
        )

        dynamics_std = float(
            np.std(frame_db)
        )

        dynamics_range = float(
            np.percentile(
                frame_db,
                95
            )
            -
            np.percentile(
                frame_db,
                5
            )
        )

    else:

        dynamics_std = 0.0
        dynamics_range = 0.0

    return {

        "sample_rate": sr,

        "duration": (
            len(audio) / sr
        ),

        "rms_db": rms_db,

        "peak_db": peak_db,

        "crest_factor_db":
            crest_factor_db,

        "integrated_lufs":
            integrated_lufs,

        "centroid_hz":
            centroid_hz,

        "rolloff_hz":
            rolloff_hz,

        "bandwidth_hz":
            bandwidth_hz,

        "stereo_width":
            stereo_width,

        "stereo_width_db":
            stereo_width_db,

        "dynamics_std":
            dynamics_std,

        "dynamics_range":
            dynamics_range,

        "bands":
            bands,
    }


# ============================================================
# BUTTERWORTH FILTER
# ============================================================

def butter_filter(
    audio,
    sr,
    cutoff,
    filter_type,
    order=2
):

    nyquist = sr / 2.0

    cutoff = min(
        cutoff,
        nyquist * 0.95
    )

    if cutoff <= 0:
        return audio

    sos = signal.butter(
        order,
        cutoff / nyquist,
        btype=filter_type,
        output="sos"
    )

    output = np.zeros_like(
        audio
    )

    for channel in range(
        audio.shape[1]
    ):

        output[:, channel] = (
            signal.sosfiltfilt(
                sos,
                audio[:, channel]
            )
        )

    return output


# ============================================================
# PEAKING EQ
# ============================================================

def peaking_eq(
    audio,
    sr,
    frequency,
    gain_db,
    q=1.0
):

    if abs(gain_db) < 0.01:
        return audio

    if frequency >= sr / 2:
        return audio

    if frequency <= 0:
        return audio

    A = 10.0 ** (
        gain_db / 40.0
    )

    omega = (
        2.0
        * np.pi
        * frequency
        / sr
    )

    alpha = (
        np.sin(omega)
        /
        (2.0 * q)
    )

    cos_omega = np.cos(
        omega
    )

    b0 = (
        1.0
        + alpha * A
    )

    b1 = (
        -2.0
        * cos_omega
    )

    b2 = (
        1.0
        - alpha * A
    )

    a0 = (
        1.0
        + alpha / A
    )

    a1 = (
        -2.0
        * cos_omega
    )

    a2 = (
        1.0
        - alpha / A
    )

    b = np.array([
        b0,
        b1,
        b2
    ]) / a0

    a = np.array([
        1.0,
        a1 / a0,
        a2 / a0
    ])

    output = np.zeros_like(
        audio
    )

    for channel in range(
        audio.shape[1]
    ):

        output[:, channel] = (
            signal.lfilter(
                b,
                a,
                audio[:, channel]
            )
        )

    return output


# ============================================================
# HIGH REGISTER PROTECTION
# ============================================================

def high_register_protection(
    audio,
    sr,
    analysis
):
    """
    Additional protection for high flute notes.

    High flute can become painful around the upper-mid
    and presence regions. This stage deliberately avoids
    boosting those frequencies.

    It also keeps saturation extremely low later in the
    chain.
    """

    centroid = analysis["centroid_hz"]

    if centroid > 2500:

        audio = peaking_eq(
            audio,
            sr,
            3500,
            -1.0,
            q=1.0
        )

        audio = peaking_eq(
            audio,
            sr,
            5000,
            -2.0,
            q=1.2
        )

    if centroid > 3200:

        audio = peaking_eq(
            audio,
            sr,
            6500,
            -2.0,
            q=1.1
        )

    if centroid > 4000:

        audio = peaking_eq(
            audio,
            sr,
            8000,
            -1.5,
            q=1.0
        )

    return audio


# ============================================================
# ADAPTIVE FLUTE EQ
# ============================================================

def adaptive_eq(
    audio,
    sr,
    analysis,
    style
):

    settings = STYLES[style]

    # --------------------------------------------------------
    # Remove subsonic rumble
    # --------------------------------------------------------

    audio = butter_filter(
        audio,
        sr,
        90,
        "highpass",
        order=2
    )

    centroid = analysis[
        "centroid_hz"
    ]

    # --------------------------------------------------------
    # Mud
    # --------------------------------------------------------

    if centroid < 1800:
        mud_cut = -2.5

    elif centroid < 2300:
        mud_cut = -1.5

    else:
        mud_cut = -0.5

    audio = peaking_eq(
        audio,
        sr,
        300,
        mud_cut,
        q=0.9
    )

    # --------------------------------------------------------
    # Body
    # --------------------------------------------------------

    audio = peaking_eq(
        audio,
        sr,
        700,
        settings["body"] * 0.6,
        q=0.8
    )

    # --------------------------------------------------------
    # Presence
    # --------------------------------------------------------

    audio = peaking_eq(
        audio,
        sr,
        2500,
        settings["presence"],
        q=0.9
    )

    # --------------------------------------------------------
    # Articulation
    # --------------------------------------------------------

    audio = peaking_eq(
        audio,
        sr,
        4200,
        settings["brightness"],
        q=1.0
    )

    # --------------------------------------------------------
    # Air
    # --------------------------------------------------------

    audio = peaking_eq(
        audio,
        sr,
        8500,
        settings["air"],
        q=0.7
    )

    # --------------------------------------------------------
    # Harshness protection
    # --------------------------------------------------------

    if centroid > 3000:

        audio = peaking_eq(
            audio,
            sr,
            5500,
            settings["harsh_cut"],
            q=1.5
        )

    if centroid > 3800:

        audio = peaking_eq(
            audio,
            sr,
            7000,
            settings["harsh_cut"] * 0.7,
            q=1.3
        )

    # --------------------------------------------------------
    # Special high flute protection
    # --------------------------------------------------------

    if style == "sweet_clear_high":

        audio = high_register_protection(
            audio,
            sr,
            analysis
        )

    return audio


# ============================================================
# REFERENCE COMPARISON
# ============================================================

def compare_reference(
    source_analysis,
    reference_analysis,
    match_amount
):

    match_amount = np.clip(
        match_amount,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Loudness
    # --------------------------------------------------------

    loudness_difference = (
        reference_analysis[
            "integrated_lufs"
        ]
        -
        source_analysis[
            "integrated_lufs"
        ]
    )

    loudness_correction = (
        loudness_difference
        * match_amount
    )

    # --------------------------------------------------------
    # Frequency bands
    # --------------------------------------------------------

    band_corrections = {}

    for band in source_analysis[
        "bands"
    ]:

        source_value = (
            source_analysis[
                "bands"
            ][band]
        )

        reference_value = (
            reference_analysis[
                "bands"
            ][band]
        )

        difference = (
            reference_value
            -
            source_value
        )

        difference = np.clip(
            difference,
            -5.0,
            5.0
        )

        band_corrections[band] = (
            difference
            * match_amount
        )

    # --------------------------------------------------------
    # Stereo
    # --------------------------------------------------------

    source_width = (
        source_analysis[
            "stereo_width"
        ]
    )

    reference_width = (
        reference_analysis[
            "stereo_width"
        ]
    )

    if source_width > 1e-6:

        width_ratio = (
            reference_width
            /
            source_width
        )

    else:

        width_ratio = 1.0

    width_ratio = np.clip(
        width_ratio,
        0.75,
        1.35
    )

    width_ratio = (
        1.0
        +
        (
            width_ratio - 1.0
        )
        * match_amount
    )

    # --------------------------------------------------------
    # Dynamics
    # --------------------------------------------------------

    source_dynamics = (
        source_analysis[
            "dynamics_range"
        ]
    )

    reference_dynamics = (
        reference_analysis[
            "dynamics_range"
        ]
    )

    dynamics_difference = (
        source_dynamics
        -
        reference_dynamics
    )

    return {

        "loudness_correction":
            loudness_correction,

        "band_corrections":
            band_corrections,

        "width_ratio":
            width_ratio,

        "source_dynamics":
            source_dynamics,

        "reference_dynamics":
            reference_dynamics,

        "dynamics_difference":
            dynamics_difference,
    }


# ============================================================
# REFERENCE EQ
# ============================================================

def apply_reference_eq(
    audio,
    sr,
    comparison
):

    corrections = comparison[
        "band_corrections"
    ]

    audio = peaking_eq(
        audio,
        sr,
        150,
        corrections["low"] * 0.65,
        q=0.7
    )

    audio = peaking_eq(
        audio,
        sr,
        350,
        corrections["low_mid"] * 0.75,
        q=0.8
    )

    audio = peaking_eq(
        audio,
        sr,
        750,
        corrections["mid"] * 0.75,
        q=0.8
    )

    audio = peaking_eq(
        audio,
        sr,
        1500,
        corrections["upper_mid"] * 0.70,
        q=0.8
    )

    audio = peaking_eq(
        audio,
        sr,
        3200,
        corrections["presence"] * 0.80,
        q=0.9
    )

    audio = peaking_eq(
        audio,
        sr,
        6500,
        corrections["brightness"] * 0.75,
        q=0.8
    )

    audio = peaking_eq(
        audio,
        sr,
        11000,
        corrections["air"] * 0.65,
        q=0.7
    )

    return audio


# ============================================================
# REFERENCE COMPRESSION ESTIMATION
# ============================================================

def reference_compression_settings(
    source_analysis,
    reference_analysis,
    match_amount
):

    source_range = (
        source_analysis[
            "dynamics_range"
        ]
    )

    reference_range = (
        reference_analysis[
            "dynamics_range"
        ]
    )

    difference = (
        source_range
        -
        reference_range
    )

    if difference > 8:

        ratio = 3.5
        threshold = -24.0

    elif difference > 5:

        ratio = 3.0
        threshold = -23.0

    elif difference > 2:

        ratio = 2.5
        threshold = -22.0

    elif difference > -2:

        ratio = 2.0
        threshold = -21.0

    else:

        ratio = 1.5
        threshold = -20.0

    ratio = (
        1.0
        +
        (
            ratio - 1.0
        )
        * match_amount
    )

    threshold = (
        -20.0
        +
        (
            threshold + 20.0
        )
        * match_amount
    )

    return {
        "ratio": ratio,
        "threshold": threshold,
        "attack_ms": 18.0,
        "release_ms": 120.0,
    }


# ============================================================
# COMPRESSOR
# ============================================================

def compressor(
    audio,
    sr,
    threshold_db=-22.0,
    ratio=2.5,
    attack_ms=18.0,
    release_ms=130.0
):

    mono = np.mean(
        audio,
        axis=1
    )

    envelope = np.abs(
        mono
    )

    attack_coeff = np.exp(
        -1.0
        /
        (
            sr
            * attack_ms
            / 1000.0
        )
    )

    release_coeff = np.exp(
        -1.0
        /
        (
            sr
            * release_ms
            / 1000.0
        )
    )

    env = np.zeros_like(
        envelope
    )

    for i in range(
        1,
        len(envelope)
    ):

        if envelope[i] > env[i - 1]:

            env[i] = (
                attack_coeff
                * env[i - 1]
                +
                (
                    1.0
                    - attack_coeff
                )
                * envelope[i]
            )

        else:

            env[i] = (
                release_coeff
                * env[i - 1]
                +
                (
                    1.0
                    - release_coeff
                )
                * envelope[i]
            )

    env_db = linear_to_db(
        env
    )

    gain_db = np.zeros_like(
        env_db
    )

    over_threshold = (
        env_db > threshold_db
    )

    gain_db[
        over_threshold
    ] = (

        threshold_db

        +

        (
            env_db[
                over_threshold
            ]
            - threshold_db
        )
        / ratio

        -

        env_db[
            over_threshold
        ]
    )

    gain = db_to_linear(
        gain_db
    )

    return (
        audio
        * gain[:, None]
    )


# ============================================================
# SATURATION
# ============================================================

def saturation(
    audio,
    amount=0.08
):

    if amount <= 0:
        return audio

    # Very conservative saturation.
    drive = 1.0 + (
        amount * 2.0
    )

    result = (
        np.tanh(
            audio * drive
        )
        /
        np.tanh(drive)
    )

    return result


# ============================================================
# REVERB
# ============================================================

def create_reverb(
    audio,
    sr,
    decay=2.2
):

    mono = np.mean(
        audio,
        axis=1
    )

    length = max(
        1,
        int(sr * decay)
    )

    impulse = np.zeros(
        length
    )

    # --------------------------------------------------------
    # Early reflections
    # --------------------------------------------------------

    reflection_times = [
        0.017,
        0.031,
        0.047,
        0.073,
        0.113,
        0.173,
        0.251,
        0.347,
    ]

    reflection_gains = [
        0.55,
        0.42,
        0.34,
        0.28,
        0.22,
        0.17,
        0.12,
        0.08,
    ]

    for time, gain in zip(
        reflection_times,
        reflection_gains
    ):

        index = int(
            sr * time
        )

        if index < length:

            impulse[index] += gain

    # --------------------------------------------------------
    # Dense tail
    # --------------------------------------------------------

    rng = np.random.default_rng(
        42
    )

    tail = rng.normal(
        0.0,
        1.0,
        length
    )

    envelope = np.exp(
        -np.linspace(
            0.0,
            6.0,
            length
        )
    )

    tail *= envelope
    tail *= 0.15

    impulse += tail

    # --------------------------------------------------------
    # Smooth
    # --------------------------------------------------------

    impulse = signal.fftconvolve(
        impulse,
        np.ones(5) / 5.0,
        mode="same"
    )

    maximum = np.max(
        np.abs(impulse)
    )

    if maximum > 0:

        impulse /= maximum

    # --------------------------------------------------------
    # Convolution
    # --------------------------------------------------------

    wet = signal.fftconvolve(
        mono,
        impulse,
        mode="full"
    )

    wet = wet[
        :len(mono)
    ]

    return np.column_stack([
        wet,
        wet
    ])


# ============================================================
# REVERB PERCENTAGE
# ============================================================

def apply_reverb(
    audio,
    sr,
    percentage,
    decay=2.2
):

    percentage = np.clip(
        percentage,
        0.0,
        100.0
    )

    if percentage <= 0:
        return audio

    wet_amount = (
        percentage / 100.0
    )

    wet = create_reverb(
        audio,
        sr,
        decay=decay
    )

    return (
        audio
        * (
            1.0
            - wet_amount
        )
        +
        wet
        * wet_amount
    )


# ============================================================
# DELAY
# ============================================================

def create_delay(
    audio,
    sr,
    delay_ms=180.0,
    feedback=0.18
):

    """
    Creates a clean stereo delay.

    The delay is intentionally subtle for flute.
    """

    delay_samples = int(
        sr
        * delay_ms
        / 1000.0
    )

    delay_samples = max(
        1,
        delay_samples
    )

    output = audio.copy()

    delayed = np.zeros_like(
        audio
    )

    for channel in range(
        audio.shape[1]
    ):

        if delay_samples < len(audio):

            delayed[
                delay_samples:,
                channel
            ] = (
                audio[
                    :-delay_samples,
                    channel
                ]
                * feedback
            )

    # Add a second, quieter echo.
    second_delay = delay_samples * 2

    if second_delay < len(audio):

        for channel in range(
            audio.shape[1]
        ):

            delayed[
                second_delay:,
                channel
            ] += (
                audio[
                    :-second_delay,
                    channel
                ]
                * feedback
                * 0.45
            )

    return output, delayed


# ============================================================
# DELAY PERCENTAGE
# ============================================================

def apply_delay(
    audio,
    sr,
    percentage
):

    percentage = np.clip(
        percentage,
        0.0,
        100.0
    )

    if percentage <= 0:
        return audio

    amount = (
        percentage / 100.0
    )

    # Flute-friendly delay.
    delay_ms = 180.0

    feedback = (
        0.12
        +
        amount * 0.18
    )

    dry, delayed = create_delay(
        audio,
        sr,
        delay_ms=delay_ms,
        feedback=feedback
    )

    return (
        dry
        +
        delayed * amount
    )


# ============================================================
# STEREO WIDENING
# ============================================================

def stereo_widen(
    audio,
    width=1.12
):

    if audio.shape[1] < 2:
        return audio

    left = audio[:, 0]
    right = audio[:, 1]

    mid = (
        left + right
    ) * 0.5

    side = (
        left - right
    ) * 0.5

    side *= width

    left_new = (
        mid + side
    )

    right_new = (
        mid - side
    )

    return np.column_stack([
        left_new,
        right_new
    ])


# ============================================================
# LIMITER
# ============================================================

def limiter(
    audio,
    ceiling_db=-1.0
):

    ceiling = db_to_linear(
        ceiling_db
    )

    peak = np.max(
        np.abs(audio)
    )

    if peak <= ceiling:
        return audio

    return (
        audio
        * ceiling
        / peak
    )


# ============================================================
# LOUDNESS NORMALIZATION
# ============================================================

def normalize_loudness(
    audio,
    sr,
    target_lufs
):

    meter = pyln.Meter(
        sr
    )

    mono = np.mean(
        audio,
        axis=1
    )

    try:

        loudness = (
            meter.integrated_loudness(
                mono
            )
        )

    except Exception:

        rms = np.sqrt(
            np.mean(mono ** 2)
        )

        loudness = (
            linear_to_db(rms)
            - 0.691
        )

    if not np.isfinite(
        loudness
    ):

        loudness = (
            target_lufs
        )

    gain_db = (
        target_lufs
        - loudness
    )

    gain_db = np.clip(
        gain_db,
        -12.0,
        12.0
    )

    gain = db_to_linear(
        gain_db
    )

    audio = (
        audio * gain
    )

    return (
        audio,
        loudness,
        gain_db
    )


# ============================================================
# REFERENCE REPORT
# ============================================================

def write_reference_report(
    output_path,
    source_analysis,
    reference_analysis,
    comparison
):

    report_path = Path(
        output_path
    ).with_suffix(
        ".reference_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "REFERENCE FLUTE MASTERING ANALYSIS\n"
        )

        f.write(
            "=" * 65
            + "\n\n"
        )

        # ----------------------------------------------------
        # Source
        # ----------------------------------------------------

        f.write(
            "SOURCE AUDIO\n"
        )

        f.write(
            "-" * 65
            + "\n"
        )

        f.write(
            f"Sample rate:       "
            f"{source_analysis['sample_rate']} Hz\n"
        )

        f.write(
            f"Duration:           "
            f"{source_analysis['duration']:.2f} sec\n"
        )

        f.write(
            f"Integrated LUFS:    "
            f"{source_analysis['integrated_lufs']:.2f}\n"
        )

        f.write(
            f"Peak:               "
            f"{source_analysis['peak_db']:.2f} dB\n"
        )

        f.write(
            f"RMS:                "
            f"{source_analysis['rms_db']:.2f} dB\n"
        )

        f.write(
            f"Crest factor:       "
            f"{source_analysis['crest_factor_db']:.2f} dB\n"
        )

        f.write(
            f"Spectral centroid:  "
            f"{source_analysis['centroid_hz']:.0f} Hz\n"
        )

        f.write(
            f"Spectral rolloff:   "
            f"{source_analysis['rolloff_hz']:.0f} Hz\n"
        )

        f.write(
            f"Stereo width:       "
            f"{source_analysis['stereo_width']:.3f}\n"
        )

        f.write(
            f"Dynamic range:      "
            f"{source_analysis['dynamics_range']:.2f} dB\n"
        )

        # ----------------------------------------------------
        # Reference
        # ----------------------------------------------------

        f.write("\n\n")

        f.write(
            "REFERENCE AUDIO\n"
        )

        f.write(
            "-" * 65
            + "\n"
        )

        f.write(
            f"Sample rate:       "
            f"{reference_analysis['sample_rate']} Hz\n"
        )

        f.write(
            f"Duration:           "
            f"{reference_analysis['duration']:.2f} sec\n"
        )

        f.write(
            f"Integrated LUFS:    "
            f"{reference_analysis['integrated_lufs']:.2f}\n"
        )

        f.write(
            f"Peak:               "
            f"{reference_analysis['peak_db']:.2f} dB\n"
        )

        f.write(
            f"RMS:                "
            f"{reference_analysis['rms_db']:.2f} dB\n"
        )

        f.write(
            f"Crest factor:       "
            f"{reference_analysis['crest_factor_db']:.2f} dB\n"
        )

        f.write(
            f"Spectral centroid:  "
            f"{reference_analysis['centroid_hz']:.0f} Hz\n"
        )

        f.write(
            f"Spectral rolloff:   "
            f"{reference_analysis['rolloff_hz']:.0f} Hz\n"
        )

        f.write(
            f"Stereo width:       "
            f"{reference_analysis['stereo_width']:.3f}\n"
        )

        f.write(
            f"Dynamic range:      "
            f"{reference_analysis['dynamics_range']:.2f} dB\n"
        )

        # ----------------------------------------------------
        # Corrections
        # ----------------------------------------------------

        f.write("\n\n")

        f.write(
            "REFERENCE MATCH CORRECTIONS\n"
        )

        f.write(
            "-" * 65
            + "\n"
        )

        f.write(
            f"Loudness correction: "
            f"{comparison['loudness_correction']:+.2f} dB\n"
        )

        f.write(
            f"Stereo multiplier:    "
            f"{comparison['width_ratio']:.3f}\n"
        )

        f.write(
            f"Source dynamics:      "
            f"{comparison['source_dynamics']:.2f} dB\n"
        )

        f.write(
            f"Reference dynamics:   "
            f"{comparison['reference_dynamics']:.2f} dB\n"
        )

        f.write("\n")

        f.write(
            "BROAD EQ CORRECTIONS\n"
        )

        f.write(
            "-" * 65
            + "\n"
        )

        for band, value in comparison[
            "band_corrections"
        ].items():

            f.write(
                f"{band:15s}: "
                f"{value:+.2f} dB\n"
            )

        f.write("\n")

        f.write(
            "NOTE:\n"
        )

        f.write(
            "These are estimates of the resulting "
            "audio characteristics. The exact plugins "
            "and original mastering settings cannot be "
            "recovered uniquely from a rendered audio file.\n"
        )

    return report_path


# ============================================================
# COMPLETE FLUTE PROCESSOR
# ============================================================

def process_flute(
    input_file,
    output_file,
    style,
    reference_file=None,
    match_amount=0.0,
    target_lufs=-14.0,
    stereo_width=1.12,
    reverb_percent=None,
    delay_percent=None,
    use_eq=True,
    use_compression=True,
    use_saturation=True,
    use_reverb=True,
    use_delay=True,
    use_stereo=True,
    use_limiter=True
):

    print()
    print("=" * 65)
    print("ADAPTIVE FLUTE STUDIO PROCESSOR")
    print("=" * 65)

    # ========================================================
    # LOAD SOURCE
    # ========================================================

    audio, sr = load_audio(
        input_file
    )

    print()
    print(
        f"Input:       {input_file}"
    )

    print(
        f"Sample rate: {sr} Hz"
    )

    print(
        f"Duration:    "
        f"{len(audio) / sr:.2f} sec"
    )

    print(
        f"Style:       {style}"
    )

    settings = STYLES[
        style
    ]

    # ========================================================
    # EFFECT PERCENTAGES
    # ========================================================

    if reverb_percent is None:

        reverb_percent = settings[
            "reverb"
        ]

    if delay_percent is None:

        delay_percent = settings[
            "delay"
        ]

    reverb_percent = np.clip(
        reverb_percent,
        0.0,
        100.0
    )

    delay_percent = np.clip(
        delay_percent,
        0.0,
        100.0
    )

    print(
        f"Reverb:      {reverb_percent:.1f}%"
    )

    print(
        f"Delay:       {delay_percent:.1f}%"
    )

    # ========================================================
    # ANALYZE SOURCE
    # ========================================================

    print()
    print(
        "Analyzing source flute..."
    )

    source_analysis = (
        analyze_reference_audio(
            audio,
            sr
        )
    )

    # ========================================================
    # REFERENCE ANALYSIS
    # ========================================================

    reference_analysis = None
    comparison = None

    if reference_file:

        print()
        print(
            "=" * 65
        )

        print(
            "REFERENCE ANALYSIS"
        )

        print(
            "=" * 65
        )

        reference_audio, reference_sr = (
            load_audio(
                reference_file
            )
        )

        print()
        print(
            "Analyzing reference..."
        )

        reference_analysis = (
            analyze_reference_audio(
                reference_audio,
                reference_sr
            )
        )

        comparison = compare_reference(
            source_analysis,
            reference_analysis,
            match_amount
        )

        print()
        print(
            f"Source LUFS:       "
            f"{source_analysis['integrated_lufs']:.2f}"
        )

        print(
            f"Reference LUFS:    "
            f"{reference_analysis['integrated_lufs']:.2f}"
        )

        print(
            f"Loudness change:   "
            f"{comparison['loudness_correction']:+.2f} dB"
        )

        print()

        print(
            f"Source centroid:   "
            f"{source_analysis['centroid_hz']:.0f} Hz"
        )

        print(
            f"Reference centroid:"
            f" {reference_analysis['centroid_hz']:.0f} Hz"
        )

        print()

        print(
            f"Source dynamics:   "
            f"{source_analysis['dynamics_range']:.2f} dB"
        )

        print(
            f"Reference dynamics:"
            f" {reference_analysis['dynamics_range']:.2f} dB"
        )

        print()

        print(
            f"Source width:      "
            f"{source_analysis['stereo_width']:.3f}"
        )

        print(
            f"Reference width:   "
            f"{reference_analysis['stereo_width']:.3f}"
        )

        # ----------------------------------------------------
        # Write report
        # ----------------------------------------------------

        report_path = write_reference_report(
            output_file,
            source_analysis,
            reference_analysis,
            comparison
        )

        print()
        print(
            "Reference report:"
        )

        print(
            report_path
        )

    # ========================================================
    # START PROCESSING
    # ========================================================

    processed = audio.copy()

    # ========================================================
    # REFERENCE EQ
    # ========================================================

    if (
        reference_analysis is not None
        and
        comparison is not None
        and
        use_eq
        and
        match_amount > 0
    ):

        print()
        print(
            "[REF] Applying reference tonal matching..."
        )

        processed = apply_reference_eq(
            processed,
            sr,
            comparison
        )

    # ========================================================
    # ADAPTIVE EQ
    # ========================================================

    if use_eq:

        print()
        print(
            "[1/9] Applying adaptive flute EQ..."
        )

        processed = adaptive_eq(
            processed,
            sr,
            analyze_audio(
                processed,
                sr
            ),
            style
        )

    else:

        print()
        print(
            "[1/9] EQ: OFF"
        )

    # ========================================================
    # COMPRESSION
    # ========================================================

    if use_compression:

        print(
            "[2/9] Applying compression..."
        )

        if (
            reference_analysis is not None
            and
            comparison is not None
            and
            match_amount > 0
        ):

            compression_settings = (
                reference_compression_settings(
                    source_analysis,
                    reference_analysis,
                    match_amount
                )
            )

            ratio = (
                compression_settings[
                    "ratio"
                ]
            )

            threshold = (
                compression_settings[
                    "threshold"
                ]
            )

            attack = (
                compression_settings[
                    "attack_ms"
                ]
            )

            release = (
                compression_settings[
                    "release_ms"
                ]
            )

            print(
                f"Reference ratio: "
                f"{ratio:.2f}:1"
            )

            print(
                f"Reference threshold: "
                f"{threshold:.1f} dB"
            )

        else:

            ratio = settings[
                "compression"
            ]

            threshold = -22.0
            attack = 18.0
            release = 130.0

        processed = compressor(
            processed,
            sr,
            threshold_db=threshold,
            ratio=ratio,
            attack_ms=attack,
            release_ms=release
        )

    else:

        print(
            "[2/9] Compression: OFF"
        )

    # ========================================================
    # SATURATION
    # ========================================================

    if use_saturation:

        print(
            "[3/9] Applying very subtle saturation..."
        )

        processed = saturation(
            processed,
            amount=settings[
                "saturation"
            ]
        )

    else:

        print(
            "[3/9] Saturation: OFF"
        )

    # ========================================================
    # REVERB
    # ========================================================

    if use_reverb and reverb_percent > 0:

        print(
            f"[4/9] Applying reverb "
            f"({reverb_percent:.1f}%)..."
        )

        processed = apply_reverb(
            processed,
            sr,
            percentage=reverb_percent,
            decay=2.2
        )

    else:

        print(
            "[4/9] Reverb: OFF"
        )

    # ========================================================
    # DELAY
    # ========================================================

    if use_delay and delay_percent > 0:

        print(
            f"[5/9] Applying delay "
            f"({delay_percent:.1f}%)..."
        )

        processed = apply_delay(
            processed,
            sr,
            percentage=delay_percent
        )

    else:

        print(
            "[5/9] Delay: OFF"
        )

    # ========================================================
    # STEREO
    # ========================================================

    if use_stereo:

        print(
            "[6/9] Applying stereo widening..."
        )

        width = stereo_width

        if (
            reference_analysis is not None
            and
            comparison is not None
            and
            match_amount > 0
        ):

            width *= (
                comparison[
                    "width_ratio"
                ]
            )

        width = np.clip(
            width,
            0.70,
            1.50
        )

        processed = stereo_widen(
            processed,
            width=width
        )

    else:

        print(
            "[6/9] Stereo widening: OFF"
        )

    # ========================================================
    # SAFETY FILTERS
    # ========================================================

    print(
        "[7/9] Applying safety filters..."
    )

    processed = butter_filter(
        processed,
        sr,
        90,
        "highpass",
        order=2
    )

    # Do not force 19 kHz when sample rate is low.
    high_cut = min(
        19000,
        sr * 0.45
    )

    processed = butter_filter(
        processed,
        sr,
        high_cut,
        "lowpass",
        order=2
    )

    # ========================================================
    # LIMITER
    # ========================================================

    if use_limiter:

        print(
            "[8/9] Applying final limiter..."
        )

        processed = limiter(
            processed,
            ceiling_db=PEAK_CEILING_DB
        )

    else:

        print(
            "[8/9] Limiter: OFF"
        )

    # ========================================================
    # TARGET LOUDNESS
    # ========================================================

    final_target_lufs = target_lufs

    if (
        reference_analysis is not None
        and
        match_amount > 0
    ):

        reference_lufs = (
            reference_analysis[
                "integrated_lufs"
            ]
        )

        final_target_lufs = (
            target_lufs
            * (
                1.0
                - match_amount
            )
            +
            reference_lufs
            * match_amount
        )

    print()
    print(
        f"Normalizing to "
        f"{final_target_lufs:.2f} LUFS..."
    )

    (
        processed,
        original_lufs,
        applied_gain
    ) = normalize_loudness(
        processed,
        sr,
        final_target_lufs
    )

    # ========================================================
    # FINAL PEAK PROTECTION
    # ========================================================

    if use_limiter:

        print(
            "[9/9] Final peak protection..."
        )

        processed = limiter(
            processed,
            ceiling_db=PEAK_CEILING_DB
        )

    # ========================================================
    # CLEAN NUMERICAL ERRORS
    # ========================================================

    processed = np.nan_to_num(
        processed,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    output_path = Path(
        output_file
    )

    if output_path.suffix.lower() != ".wav":

        output_path = (
            output_path.with_suffix(
                ".wav"
            )
        )

    print()
    print(
        "Saving 24-bit WAV..."
    )

    sf.write(
        str(output_path),
        processed,
        sr,
        subtype="PCM_24"
    )

    # ========================================================
    # FINAL ANALYSIS
    # ========================================================

    final_analysis = (
        analyze_reference_audio(
            processed,
            sr
        )
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 65)
    print("PROCESSING COMPLETE")
    print("=" * 65)

    print()
    print(
        f"Output:        {output_path}"
    )

    print(
        f"Style:         {style}"
    )

    print(
        f"Final LUFS:    "
        f"{final_analysis['integrated_lufs']:.2f}"
    )

    print(
        f"Final peak:    "
        f"{final_analysis['peak_db']:.2f} dB"
    )

    print(
        f"Final RMS:     "
        f"{final_analysis['rms_db']:.2f} dB"
    )

    print(
        f"Final centroid:"
        f" {final_analysis['centroid_hz']:.0f} Hz"
    )

    print(
        f"Final width:   "
        f"{final_analysis['stereo_width']:.3f}"
    )

    print(
        f"Gain applied:  "
        f"{applied_gain:+.2f} dB"
    )

    # ========================================================
    # EFFECT STATUS
    # ========================================================

    print()
    print("EFFECTS")
    print("-" * 40)

    print(
        f"EQ:            "
        f"{'ON' if use_eq else 'OFF'}"
    )

    print(
        f"Compression:   "
        f"{'ON' if use_compression else 'OFF'}"
    )

    print(
        f"Saturation:    "
        f"{'ON' if use_saturation else 'OFF'}"
    )

    print(
        f"Reverb:        "
        f"{reverb_percent:.1f}%"
        if use_reverb
        else
        "Reverb:        OFF"
    )

    print(
        f"Delay:         "
        f"{delay_percent:.1f}%"
        if use_delay
        else
        "Delay:         OFF"
    )

    print(
        f"Stereo:        "
        f"{'ON' if use_stereo else 'OFF'}"
    )

    print(
        f"Limiter:       "
        f"{'ON' if use_limiter else 'OFF'}"
    )

    if reference_file:

        print()
        print(
            f"Reference:     {reference_file}"
        )

        print(
            f"Reference match:"
            f" {match_amount * 100:.0f}%"
        )

    print()
    print(
        "Done."
    )


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Professional adaptive flute processor "
            "with sweet/clear profiles and reference matching."
        )
    )

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    parser.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_INPUT,
        help=(
            "Input audio file "
            "(WAV/MP3/M4A/FLAC/AAC/OGG/OPUS)"
        )
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output WAV file"
    )

    # --------------------------------------------------------
    # Style
    # --------------------------------------------------------

    parser.add_argument(
        "--style",
        choices=[
            "warm",
            "natural",
            "bright",
            "cinematic",
            "studio",
            "sweet_clear",
            "sweet_clear_high",
        ],
        default=DEFAULT_STYLE,
        help=(
            "Flute sound profile"
        )
    )

    # --------------------------------------------------------
    # Reference
    # --------------------------------------------------------

    parser.add_argument(
        "--reference",
        default=None,
        help=(
            "Reference audio file to analyze "
            "and match"
        )
    )

    parser.add_argument(
        "--match",
        type=float,
        default=0.0,
        help=(
            "Reference matching strength "
            "from 0 to 100"
        )
    )

    # --------------------------------------------------------
    # Loudness
    # --------------------------------------------------------

    parser.add_argument(
        "--lufs",
        type=float,
        default=DEFAULT_TARGET_LUFS,
        help=(
            "Normal target loudness "
            "(default: -14 LUFS)"
        )
    )

    # --------------------------------------------------------
    # Stereo
    # --------------------------------------------------------

    parser.add_argument(
        "--width",
        type=float,
        default=DEFAULT_STEREO_WIDTH,
        help=(
            "Stereo width "
            "(default: 1.12)"
        )
    )

    # --------------------------------------------------------
    # REVERB PERCENTAGE
    # --------------------------------------------------------

    parser.add_argument(
        "--reverb",
        type=float,
        default=None,
        help=(
            "Reverb amount from 0 to 100 percent. "
            "0 = dry, 100 = fully wet. "
            "If omitted, style default is used."
        )
    )

    # --------------------------------------------------------
    # DELAY PERCENTAGE
    # --------------------------------------------------------

    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help=(
            "Delay amount from 0 to 100 percent. "
            "0 = off. "
            "If omitted, style default is used."
        )
    )

    # --------------------------------------------------------
    # Effect switches
    # --------------------------------------------------------

    parser.add_argument(
        "--eq",
        choices=[
            "on",
            "off"
        ],
        default="on",
        help="Enable/disable EQ"
    )

    parser.add_argument(
        "--compression",
        choices=[
            "on",
            "off"
        ],
        default="on",
        help="Enable/disable compression"
    )

    parser.add_argument(
        "--saturation",
        choices=[
            "on",
            "off"
        ],
        default="on",
        help="Enable/disable saturation"
    )

    parser.add_argument(
        "--reverb-switch",
        choices=[
            "on",
            "off"
        ],
        default="on",
        help="Enable/disable reverb"
    )

    parser.add_argument(
        "--delay-switch",
        choices=[
            "on",
            "off"
        ],
        default="on",
        help="Enable/disable delay"
    )

    parser.add_argument(
        "--stereo",
        choices=[
            "on",
            "off"
        ],
        default="on",
        help="Enable/disable stereo widening"
    )

    parser.add_argument(
        "--limiter",
        choices=[
            "on",
            "off"
        ],
        default="on",
        help="Enable/disable limiter"
    )

    args = parser.parse_args()

    # ========================================================
    # VALIDATE MATCH
    # ========================================================

    match_amount = np.clip(
        args.match / 100.0,
        0.0,
        1.0
    )

    # ========================================================
    # VALIDATE REVERB
    # ========================================================

    if args.reverb is not None:

        reverb_percent = np.clip(
            args.reverb,
            0.0,
            100.0
        )

    else:

        reverb_percent = None

    # ========================================================
    # VALIDATE DELAY
    # ========================================================

    if args.delay is not None:

        delay_percent = np.clip(
            args.delay,
            0.0,
            100.0
        )

    else:

        delay_percent = None

    # ========================================================
    # PROCESS
    # ========================================================

    process_flute(

        input_file=args.input,

        output_file=args.output,

        style=args.style,

        reference_file=args.reference,

        match_amount=match_amount,

        target_lufs=args.lufs,

        stereo_width=args.width,

        reverb_percent=reverb_percent,

        delay_percent=delay_percent,

        use_eq=(
            args.eq == "on"
        ),

        use_compression=(
            args.compression == "on"
        ),

        use_saturation=(
            args.saturation == "on"
        ),

        use_reverb=(
            args.reverb_switch == "on"
        ),

        use_delay=(
            args.delay_switch == "on"
        ),

        use_stereo=(
            args.stereo == "on"
        ),

        use_limiter=(
            args.limiter == "on"
        )
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
