from __future__ import annotations

import sys
from pathlib import Path

# Allow direct execution without pre-setting PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_LEROBOT_ROOT = _REPO_ROOT / "external" / "lerobot"
for _p in (_LEROBOT_ROOT, _REPO_ROOT):
    _s = str(_p)
    if _p.exists() and _s not in sys.path:
        sys.path.insert(0, _s)
