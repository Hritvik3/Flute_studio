"""
FastAPI application: REST API + static frontend + media serving.
"""

from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import dsp, library, separation
from app.config import (
    ALL_DATA_DIRS,
    MAX_UPLOAD_BYTES,
    PORT,
    ensure_dirs,
)
from app.jobs import JOBS

ensure_dirs()

app = FastAPI(title="Flute Studio", version="1.0.0")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


# ============================================================
# Health / meta
# ============================================================

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "device": separation.device_name(),
        "models": [
            {"id": mid, "label": m["label"], "sources": m["sources"]}
            for mid, m in separation.MODELS.items()
        ],
        "styles": dsp.STYLES,
        "defaults": {
            "target_lufs": dsp.DEFAULT_TARGET_LUFS,
            "stereo_width": dsp.DEFAULT_STEREO_WIDTH,
            "ceiling_db": dsp.DEFAULT_PEAK_CEILING_DB,
        },
    }


@app.get("/health")
def health_alias():
    """Alias some tools poll automatically."""
    return {"status": "ok"}


# ============================================================
# Library
# ============================================================

@app.get("/api/library")
def api_list_library(kind: str | None = None):
    return library.list_files(kind)


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large")
    try:
        record = library.save_upload(file.filename or "audio.wav", data)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    # Probe the file so broken uploads fail fast with a clear message
    try:
        record["analysis"] = library.analyze_file(record["path"])
    except Exception as exc:  # noqa: BLE001
        library.delete_file(record["id"])
        raise HTTPException(
            400, f"Could not decode audio: {exc}"
        ) from exc
    return record


@app.delete("/api/library/{file_id}")
def api_delete_file(file_id: str):
    if not library.delete_file(file_id):
        raise HTTPException(404, "File not found")
    return {"deleted": file_id}


@app.get("/api/analysis/{file_id}")
def api_analysis(file_id: str):
    record = library.get_file(file_id)
    if record is None:
        raise HTTPException(404, "File not found")
    try:
        return library.analyze_file(record["path"])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Analysis failed: {exc}") from exc


# ============================================================
# Jobs
# ============================================================

@app.get("/api/jobs")
def api_list_jobs(limit: int = 50):
    return JOBS.list(limit)


@app.get("/api/jobs/{job_id}")
def api_get_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job.to_dict()


@app.post("/api/jobs/{job_id}/cancel")
def api_cancel_job(job_id: str):
    job = JOBS.cancel(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job.to_dict()


# ============================================================
# Separation
# ============================================================

class SeparateRequest(BaseModel):
    file_id: str
    model: str = separation.DEFAULT_MODEL
    name: str | None = None


@app.post("/api/separate")
def api_separate(req: SeparateRequest, background: BackgroundTasks):
    record = library.get_file(req.file_id)
    if record is None:
        raise HTTPException(404, "File not found")
    if req.model not in separation.MODELS:
        raise HTTPException(400, f"Unknown model: {req.model}")

    path = record["path"]
    name = req.name or Path(record["name"]).stem

    def run(job):
        manifest = separation.separate_file(path, job, req.model)
        set_record = library.register_stem_set(manifest, name=name)
        return {"stemset": set_record}

    job = JOBS.submit("separate", f"Separate {name}", run)
    return job.to_dict()


# ============================================================
# Mastering
# ============================================================

class MasterRequest(BaseModel):
    file_id: str
    style: str = "studio"
    reference_id: str | None = None
    match_percent: float = Field(0, ge=0, le=100)
    target_lufs: float = Field(dsp.DEFAULT_TARGET_LUFS, ge=-30, le=0)
    stereo_width: float = Field(dsp.DEFAULT_STEREO_WIDTH, ge=0.5, le=1.6)
    reverb_percent: float | None = Field(None, ge=0, le=100)
    delay_percent: float | None = Field(None, ge=0, le=100)
    ceiling_db: float = Field(dsp.DEFAULT_PEAK_CEILING_DB, ge=-12, le=0)
    use_eq: bool = True
    use_compression: bool = True
    use_saturation: bool = True
    use_reverb: bool = True
    use_delay: bool = True
    use_stereo: bool = True
    use_limiter: bool = True
    output_name: str | None = None


@app.post("/api/master")
def api_master(req: MasterRequest):
    record = library.get_file(req.file_id)
    if record is None:
        raise HTTPException(404, "Source file not found")
    if req.style not in dsp.STYLES:
        raise HTTPException(400, f"Unknown style: {req.style}")

    reference_record = None
    if req.reference_id:
        reference_record = library.get_file(req.reference_id)
        if reference_record is None:
            raise HTTPException(404, "Reference file not found")

    path = record["path"]
    ref_path = reference_record["path"] if reference_record else None
    from app.config import MASTERS_DIR

    output_name = req.output_name or Path(record["name"]).stem
    options = req.model_dump()

    def run(job):
        job.set_progress(0.02, "Loading audio")
        audio, sr = dsp.load_audio(path)

        reference_audio = None
        if ref_path:
            job.set_progress(0.05, "Loading reference")
            reference_audio = dsp.load_audio(ref_path)

        opts = {
            "style": options["style"],
            "reference_audio": reference_audio,
            "match_amount": options["match_percent"] / 100.0,
            "target_lufs": options["target_lufs"],
            "stereo_width": options["stereo_width"],
            "reverb_percent": options["reverb_percent"],
            "delay_percent": options["delay_percent"],
            "ceiling_db": options["ceiling_db"],
            "use_eq": options["use_eq"],
            "use_compression": options["use_compression"],
            "use_saturation": options["use_saturation"],
            "use_reverb": options["use_reverb"],
            "use_delay": options["use_delay"],
            "use_stereo": options["use_stereo"],
            "use_limiter": options["use_limiter"],
        }

        processed, report = dsp.master_chain(
            audio, sr, opts,
            on_progress=lambda f, s: job.set_progress(0.1 + 0.85 * f, s),
            on_log=job.log,
        )

        job.raise_if_cancelled()
        job.set_progress(0.97, "Writing WAV")
        out_path = MASTERS_DIR / f"{output_name}_{job.id}.wav"
        dsp.save_wav(out_path, processed, sr)

        rec = library.register_file(
            out_path,
            kind="master",
            name=f"{output_name} ({options['style']})",
            extra={"report": report, "source_id": req.file_id},
        )
        return {"master": rec, "report": report}

    job = JOBS.submit("master", f"Master {output_name}", run)
    return job.to_dict()


# ============================================================
# Mixing
# ============================================================

class MixTrack(BaseModel):
    file_id: str
    gain_db: float = Field(0, ge=-60, le=12)
    pan: float = Field(0, ge=-1, le=1)
    mute: bool = False
    solo: bool = False


class MixRequest(BaseModel):
    tracks: list[MixTrack]
    master_gain_db: float = Field(0, ge=-24, le=12)
    target_lufs: float | None = Field(None, ge=-30, le=0)
    ceiling_db: float = Field(dsp.DEFAULT_PEAK_CEILING_DB, ge=-12, le=0)
    limiter: bool = True
    output_name: str | None = None


@app.post("/api/mix")
def api_mix(req: MixRequest):
    if not req.tracks:
        raise HTTPException(400, "Add at least one track")

    resolved = []
    for t in req.tracks:
        rec = library.get_file(t.file_id)
        if rec is None:
            raise HTTPException(404, f"Track not found: {t.file_id}")
        resolved.append((rec, t))

    from app.config import MIXES_DIR

    def run(job):
        job.set_progress(0.02, "Loading tracks")
        tracks = []
        for i, (rec, t) in enumerate(resolved):
            job.raise_if_cancelled()
            audio, sr = dsp.load_audio(rec["path"])
            tracks.append(
                {
                    "audio": audio,
                    "sr": sr,
                    "name": rec["name"],
                    "gain_db": t.gain_db,
                    "pan": t.pan,
                    "mute": t.mute,
                    "solo": t.solo,
                }
            )
            job.log(f"Loaded: {rec['name']}")
            job.set_progress(
                0.02 + 0.15 * (i + 1) / len(resolved), "Loading tracks"
            )

        mix, sr, report = dsp.render_mix(
            tracks,
            master={
                "gain_db": req.master_gain_db,
                "target_lufs": req.target_lufs,
                "ceiling_db": req.ceiling_db,
                "limiter": req.limiter,
            },
            on_progress=lambda f, s: job.set_progress(
                0.17 + 0.8 * f, s
            ),
            on_log=job.log,
        )

        job.raise_if_cancelled()
        out_name = req.output_name or "mix"
        out_path = MIXES_DIR / f"{out_name}_{job.id}.wav"
        job.set_progress(0.97, "Writing WAV")
        dsp.save_wav(out_path, mix, sr)

        rec = library.register_file(
            out_path, kind="mix", name=out_name, extra={"report": report}
        )
        return {"mix": rec, "report": report}

    job = JOBS.submit("mix", "Render mix", run)
    return job.to_dict()


# ============================================================
# Media + frontend
# ============================================================

for _dir in ALL_DATA_DIRS:
    app.mount(
        f"/media/{_dir.name}",
        StaticFiles(directory=_dir),
        name=f"media-{_dir.name}",
    )


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


def main():
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=PORT)


if __name__ == "__main__":
    main()
