# 🎐 Flute Studio

**Split, mix and master audio in your browser — a local, GPU-accelerated audio
studio built on Demucs and a dedicated adaptive flute mastering engine.**

[![License: MIT](https://img.shields.io/badge/License-MIT-4cc9a6.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-6c8cff.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688.svg)](https://fastapi.tiangolo.com/)
[![Demucs](https://img.shields.io/badge/Demucs-stem%20separation-ff6b6b.svg)](https://github.com/facebookresearch/demucs)
[![CUDA](https://img.shields.io/badge/GPU-CUDA%20optional-f0a26c.svg)](https://pytorch.org/)

Flute Studio turns a full song into stems, lets you rebalance them on a
per-track mixer, and runs a professional adaptive mastering chain — all from a
clean web UI, entirely on your own machine. No cloud, no accounts, no upload
limits.

It was built to answer a concrete problem: *how do I take a professional
recording, pull out the parts I want, and make my own flute sit in that mix
like it belongs there?*

---

## Table of contents

- [Highlights](#highlights)
- [Quick start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [Architecture](#architecture)
- [REST API](#rest-api)
- [Performance notes](#performance-notes)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Highlights

### 🔀 Separate
- **Demucs** `htdemucs_6s` (drums · bass · other · vocals · guitar · piano) or
  `htdemucs` (4 stems)
- Uses **CUDA automatically** when available, transparent CPU fallback
- Real progress reporting, live logs and **cancellable** background jobs
- Stems exported as 24-bit WAV and added to the library

### 🎛 Mix
- Any library file (stem, upload, previous mix/master) as a mixer track
- Per-track **gain · pan · mute · solo** (constant-power pan law)
- Master bus with gain, **LUFS normalization** and a peak limiter
- One-click *send stems to mixer* from the separation result

### 🎚 Master
The full adaptive chain from `flute_processor.py`:

```
adaptive EQ → compression → saturation → reverb → delay → stereo widening
→ safety filters → limiter → LUFS normalization → final peak protection
```

- 7 profiles: `warm`, `natural`, `bright`, `cinematic`, `studio`,
  `sweet_clear`, `sweet_clear_high` (the last one tames harsh high-register
  flute)
- **Reference matching** — analyze any professional recording and transfer its
  tonal balance, dynamics, stereo width and loudness at 0–100 % strength
- Fine controls for every stage, plus a before/after analysis table
  (LUFS, peak, crest factor, dynamic range, spectral centroid, stereo width)

### 📚 Library & jobs
- Drag-and-drop uploads: WAV · MP3 · M4A · FLAC · AAC · OGG · OPUS
- Every file analyzed on upload (loudness, spectrum, waveform)
- Jobs tab with progress bars, full logs and cancellation

---

## Quick start

```bash
git clone https://github.com/<you>/flute-studio.git
cd flute-studio
pip install -r requirements.txt
python -m app.main
```

Open **http://127.0.0.1:8000** — done.

> **Windows:** double-click `run.bat`. **macOS / Linux:** `./run.sh`.

---

## Installation

### Requirements

| Requirement | Notes |
|---|---|
| **Python 3.10+** | Tested on 3.11 |
| **FFmpeg** | On your `PATH`; needed for MP3/M4A/AAC decoding |
| **NVIDIA GPU** *(optional)* | CUDA build of PyTorch — separation is ~10–20× faster |

```bash
# 1. FFmpeg (verify it's on PATH)
ffmpeg -version

# 2. Python dependencies
pip install -r requirements.txt
```

<details>
<summary>Installing PyTorch with CUDA support</summary>

Flute Studio auto-detects the GPU. To get the CUDA build:

```bash
# pick the command matching your CUDA version: https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

CPU-only machines work fine — just expect separation of a 4-minute song to
take 10–20 minutes instead of ~1 minute.

</details>

---

## Usage

A typical flute-in-a-song workflow:

1. **Library** → upload your song *and* your raw flute recording.
2. **Separate** → split the song into stems → *Send stems to Mixer*.
3. **Mix** → set stem levels and panning, mute what you don't need, add your
   flute as a track, render the mix.
4. **Master** → polish the flute stem (or the final mix) with a style profile,
   optionally matching a professional reference recording.

All output is rendered to **24-bit WAV** in `data/` — ready to drop into any
DAW (Ableton, FL Studio, Logic, Reaper…).

> **Tip:** when mastering a flute stem that will be placed into an existing
> mix, disable reverb and delay — handle ambience in your DAW instead, so the
> stem stays clean and editable.

---

## Configuration

Environment variables (all optional):

| Variable | Default | Purpose |
|---|---|---|
| `FLUTE_STUDIO_PORT` | `8000` | Server port |
| `FLUTE_STUDIO_HOST` | `127.0.0.1` | Bind address (`0.0.0.0` to expose on LAN) |
| `FLUTE_STUDIO_DATA` | `./data` | Audio storage directory |
| `FLUTE_STUDIO_MAX_UPLOAD_MB` | `500` | Upload size limit |
| `FLUTE_PROCESSOR_PATH` | *(auto)* | Explicit path to a `flute_processor.py` engine |

Example (Windows):

```bat
set FLUTE_STUDIO_PORT=8001
run.bat
```

---

## Project structure

```text
flute-studio/
├── app/
│   ├── main.py         FastAPI app: REST API + static frontend + media serving
│   ├── config.py       paths, limits, supported formats
│   ├── dsp.py          mastering chain, fast compressor, mixer
│   ├── separation.py   Demucs wrapper (Python API, progress callbacks)
│   ├── jobs.py         background job manager (threads, logs, cancel)
│   └── library.py      upload registry, analysis, waveform/spectrum data
├── frontend/
│   ├── index.html      single-page UI (no build step, no framework)
│   ├── style.css
│   └── app.js
├── flute_processor.py  the adaptive flute mastering engine (DSP source of truth)
├── data/               audio storage — created on first run, git-ignored
├── requirements.txt
├── run.bat / run.sh
└── LICENSE
```

---

## Architecture

```text
┌───────────────┐   HTTP/JSON    ┌──────────────────────────┐
│   Browser UI  │ ─────────────▶ │  FastAPI (app/main.py)   │
│  (vanilla JS) │ ◀─── poll ──── │  routes · validation     │
└───────────────┘                └───────┬──────────┬───────┘
                                         │          │
                              ┌──────────▼──┐  ┌────▼─────────────┐
                              │ JobManager  │  │ Library registry │
                              │ threads ·   │  │ uploads · stems ·│
                              │ progress ·  │  │ mixes · masters  │
                              │ cancel      │  └──────────────────┘
                              └──────┬──────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
             ┌────────────┐  ┌─────────────┐  ┌──────────────┐
             │ Demucs     │  │ Mixer       │  │ Master chain │
             │ (torch,    │  │ gain/pan/   │  │ flute_       │
             │ GPU/CPU)   │  │ mute/solo + │  │ processor.py │
             │            │  │ master bus  │  │ DSP          │
             └────────────┘  └─────────────┘  └──────────────┘
```

Design decisions worth knowing:

- **One DSP source of truth.** `flute_processor.py` is imported directly —
  the web app adds job control, progress and a mixer around it, never a fork of
  the audio code. A bundled copy ships in the repo; a copy next to the app
  folder (or `FLUTE_PROCESSOR_PATH`) takes precedence if present.
- **Jobs, not requests.** Separation and rendering run on worker threads with
  polled progress, so long renders never block the UI and can be cancelled.
- **The compressor runs at control rate.** Same gain law as the per-sample
  reference implementation, but the envelope follower ticks at ~1 kHz —
  ~1000× fewer iterations, which is what makes full-song jobs interactive.
- **No build step.** The frontend is three static files served by FastAPI.

---

## REST API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Device, models, styles, defaults |
| `GET` | `/api/library` | List files (`?kind=upload\|stem\|stemset\|mix\|master`) |
| `POST` | `/api/upload` | Upload audio (multipart) |
| `DELETE` | `/api/library/{id}` | Delete a file |
| `GET` | `/api/analysis/{id}` | Loudness, spectrum, waveform peaks |
| `POST` | `/api/separate` | `{file_id, model}` → separation job |
| `POST` | `/api/mix` | `{tracks:[{file_id,gain_db,pan,mute,solo}], …}` → mix job |
| `POST` | `/api/master` | `{file_id, style, reference_id, match_percent, …}` → master job |
| `GET` | `/api/jobs` · `/api/jobs/{id}` | Job status / progress / logs |
| `POST` | `/api/jobs/{id}/cancel` | Cancel a running job |

Interactive docs are served at **`/docs`** (Swagger UI) when the server runs.

```bash
# headless example: upload + master
curl -F "file=@flute.m4a" http://127.0.0.1:8000/api/upload
curl -X POST http://127.0.0.1:8000/api/master \
  -H "Content-Type: application/json" \
  -d '{"file_id":"<id>","style":"sweet_clear","target_lufs":-14}'
```

---

## Performance notes

- Rendering is in-memory; tracks longer than ~15 minutes use a lot of RAM.
- Separations are serialized on the GPU — queued jobs wait their turn.
- Cancellation takes effect between stages / Demucs segments.
- This is an automated starting point, not a replacement for critical
  listening — always check the result on headphones, speakers and in mono.

### Troubleshooting

| Symptom | Fix |
|---|---|
| `Could not decode the M4A/AAC file` | Install FFmpeg and add it to `PATH` |
| Separation is very slow | You're on CPU — install a CUDA build of torch |
| `File too large` | Raise `FLUTE_STUDIO_MAX_UPLOAD_MB` |
| Port already in use | Set `FLUTE_STUDIO_PORT=8001` |
| Job failed | **Jobs** tab → expand the log for the full traceback |

---

## Contributing

Contributions are welcome! In short:

1. Fork and create a feature branch (`git checkout -b feature/my-feature`)
2. Keep changes focused; match the existing code style
3. Test manually against the UI (upload → separate → mix → master)
4. Open a pull request describing the change and why

Please open an issue first for large changes (new effects, UI redesign,
alternative separation models).

---

## License

Distributed under the [MIT License](LICENSE). Free to use, modify and
redistribute — including commercially.

---

## Acknowledgements

- [Demucs](https://github.com/facebookresearch/demucs) by Meta AI — the source
  separation model
- [FastAPI](https://fastapi.tiangolo.com/), [librosa](https://librosa.org/),
  [soundfile](https://github.com/bastibe/python-soundfile),
  [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) — the audio/web
  stack underneath

---

Made 🎐 for flute players who want their recordings to sit in a professional mix.
