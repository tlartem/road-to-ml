"""Apply Feast feature definitions and materialize to online store."""

import logging
import subprocess
import sys
from datetime import datetime

log = logging.getLogger(__name__)

FEAST_REPO = "/app/feast"


def _run_cmd(cmd):
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=FEAST_REPO, capture_output=True, text=True)
    if result.stdout:
        log.info(result.stdout)
    if result.returncode != 0:
        log.error("STDERR: %s", result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def run():
    log.info("Applying Feast feature definitions...")
    _run_cmd(["/app/.venv/bin/feast", "apply"])

    end_date = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    log.info("Materializing to online store (end_date=%s)...", end_date)
    _run_cmd(["/app/.venv/bin/feast", "materialize-incremental", end_date])

    log.info("Feast materialization complete!")
