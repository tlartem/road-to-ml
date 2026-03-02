"""Apply Feast feature definitions and materialize to online store.

Runs after feature_engineering.py has saved zone_stats.parquet to PVC.
"""

import logging
import subprocess
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

FEAST_REPO = "/app/feast"


def run(cmd):
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=FEAST_REPO, capture_output=True, text=True)
    if result.stdout:
        log.info(result.stdout)
    if result.returncode != 0:
        log.error("STDERR: %s", result.stderr)
        sys.exit(1)


def main():
    # Apply feature definitions (create/update registry)
    log.info("Applying Feast feature definitions...")
    run(["/app/.venv/bin/feast", "apply"])

    # Materialize to online store
    end_date = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    log.info("Materializing to online store (end_date=%s)...", end_date)
    run(["/app/.venv/bin/feast", "materialize-incremental", end_date])

    log.info("Feast materialization complete!")


if __name__ == "__main__":
    main()
