"""
Freeze guard for the ABIDE II preregistration.

Scoring ABIDE II participants before the analysis plan is frozen would defeat
the plan, so evaluation checks the document's recorded frozen commit first.
This stops accidental early analysis; git history is the audit trail for
anything else.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Optional

PREREG_PATH = Path("preregistrations/2026-09-28-abide2-external-validation.md")
_FROZEN_RE = re.compile(r"^\|\s*Frozen commit\s*\|\s*`?([^`|\s]+)`?\s*\|", re.MULTILINE)
UNFROZEN = "UNFROZEN"
MIN_SHA_LENGTH = 7


class PreregistrationNotFrozen(RuntimeError):
    pass


def frozen_sha(text: str) -> Optional[str]:
    """The frozen commit recorded in the document, or None while unfrozen."""
    match = _FROZEN_RE.search(text)
    if not match or match.group(1) == UNFROZEN:
        return None
    return match.group(1)


def _commit_exists(sha: str, repo: Path) -> bool:
    proc = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
        cwd=str(repo), capture_output=True,
    )
    return proc.returncode == 0


def require_frozen(claimed_sha: Optional[str], repo: Path,
                   prereg_path: Path = PREREG_PATH) -> str:
    """Return the frozen SHA, or raise if the plan is unfrozen or mismatched.

    ``claimed_sha`` is what the operator passes on the command line; requiring
    it forces them to look at which plan version they are running under.
    """
    recorded = frozen_sha((repo / prereg_path).read_text(encoding="utf-8"))
    if recorded is None:
        raise PreregistrationNotFrozen(
            f"{prereg_path} is not frozen. Follow its §0 freeze procedure first."
        )
    if not claimed_sha or len(claimed_sha) < MIN_SHA_LENGTH:
        raise PreregistrationNotFrozen(
            f"pass --prereg-sha (at least {MIN_SHA_LENGTH} characters); "
            f"the plan is frozen at {recorded}"
        )
    if not (recorded.startswith(claimed_sha) or claimed_sha.startswith(recorded)):
        raise PreregistrationNotFrozen(
            f"--prereg-sha {claimed_sha} does not match the frozen commit {recorded}"
        )
    if not _commit_exists(recorded, repo):
        raise PreregistrationNotFrozen(
            f"frozen commit {recorded} is not in this repository's history"
        )
    return recorded
