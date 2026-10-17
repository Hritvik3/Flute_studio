"""
Flute Studio App package.

Resolves the DSP engine (flute_processor.py) and puts it on sys.path.
Search order:

1. FLUTE_PROCESSOR_PATH env var (explicit override: file or directory)
2. Workspace root (the folder *above* the repo, for dev layouts where the
   app lives next to a working copy of flute_processor.py)
3. Repo root (the bundled copy shipped with this repository)

so a fresh clone works out of the box.
"""

import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent        # .../app
REPO_ROOT = APP_DIR.parent                       # .../flute-studio-app (repo root)
WORKSPACE_ROOT = REPO_ROOT.parent                # original workspace layout

_candidates = []

_override = os.environ.get("FLUTE_PROCESSOR_PATH")
if _override:
    _override = Path(_override)
    _candidates.append(
        _override if _override.is_dir() else _override.parent
    )

_candidates.extend([WORKSPACE_ROOT, REPO_ROOT])

for _root in _candidates:
    if (_root / "flute_processor.py").exists():
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        break
