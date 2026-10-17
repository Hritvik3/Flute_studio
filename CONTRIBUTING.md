# Contributing to Flute Studio

Thanks for your interest in contributing! This project is a local,
browser-based audio studio: Demucs for separation, a mixer, and the adaptive
`flute_processor.py` mastering engine. Keep changes focused and they'll be
easy to review.

## Getting set up

```bash
git clone https://github.com/<you>/flute-studio.git
cd flute-studio
pip install -r requirements.txt
python -m app.main
```

Open http://127.0.0.1:8000. You'll need **FFmpeg** on your `PATH`; a CUDA GPU
is optional but makes separation ~10–20× faster.

## Ground rules

- **Python:** 3.10+, standard library where reasonable, type hints on new
  public functions.
- **Frontend:** plain HTML/CSS/JS — no build step, no framework. Keep it that
  way unless there's a strong reason not to.
- **DSP:** `flute_processor.py` is the single source of truth for audio
  processing. The web app should *orchestrate* it, never fork its audio code.
  If you change processing behavior, change it there (and update the bundled
  copy in the repo root — see below).
- **Long work belongs in jobs.** Anything that can take more than a second or
  two (rendering, separation, analysis of long files) must run through the
  job manager (`app/jobs.py`) with progress reporting, not inside a request.
- **No audio in git.** `data/` and all audio extensions are git-ignored —
  please keep it that way.

## Making a change

1. Fork and create a branch:

   ```bash
   git checkout -b feature/my-feature
   ```

2. Keep each PR to one logical change. Large features (new effects, extra
   separation models, UI redesigns) — please open an issue first to discuss.

3. Test manually against the UI before submitting. The core path to verify:

   - upload a track (Library)
   - separate it (Separate tab)
   - send stems to the mixer and render a mix (Mix tab)
   - master a track with a style, and with a reference (Master tab)
   - check the Jobs tab shows progress/logs and that cancel works

4. If you touch the API, update the table in `README.md`.

## Pull request checklist

- [ ] Works end-to-end in the UI (separate → mix → master)
- [ ] No large files or audio committed (`git status` is clean of `data/`)
- [ ] README/API docs updated if behavior changed
- [ ] Clear description of *what* changed and *why*

## Reporting bugs

Open an issue including:

- OS and Python version (`python --version`)
- Whether CUDA is available (the UI header shows GPU/CPU)
- Exact steps to reproduce
- The **Jobs tab log** for any failed job — it contains the full traceback

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE) of this project.
