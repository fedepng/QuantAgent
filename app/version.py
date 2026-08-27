from __future__ import annotations

import os
import subprocess
from pathlib import Path

BUILD_VERSION = "2.0.2"


def code_version(root: Path) -> dict[str, str]:
    configured = os.getenv("QUANTAGENT_BUILD_COMMIT", "").strip()
    if configured:
        return {"code_version": configured, "code_version_source": "build"}
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        if value:
            return {"code_version": value, "code_version_source": "git"}
    except (OSError, subprocess.SubprocessError):
        pass
    return {"code_version": f"{BUILD_VERSION}+unknown", "code_version_source": "unknown"}
