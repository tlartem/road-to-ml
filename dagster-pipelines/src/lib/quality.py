"""Data quality validation using Pandera.

Reads check definitions from checks/*.yaml, builds Pandera schemas,
validates pandas DataFrames, and pushes metrics to VictoriaMetrics.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandera as pa
import requests
import yaml

log = logging.getLogger(__name__)

VM_URL = os.environ.get("VM_URL", "http://victoriametrics.monitoring.svc.cluster.local:8428")
CHECKS_DIR = Path(__file__).parent.parent.parent / "checks"


@dataclass
class QualityResult:
    """Result of a data quality check."""
    table: str
    success: bool
    total_checks: int = 0
    passed: int = 0
    failed: int = 0
    rows: int = 0
    failures: list = field(default_factory=list)


def _build_schema(spec, df_columns):
    """Build a Pandera DataFrameSchema from a YAML spec dict."""
    columns = {}
    for col_name, rules in spec.get("columns", {}).items():
        if col_name not in df_columns:
            continue
        checks = []
        nullable = rules.get("nullable", True)
        unique = rules.get("unique", False)
        if "min" in rules and "max" in rules:
            checks.append(pa.Check.in_range(rules["min"], rules["max"]))
        elif "min" in rules:
            checks.append(pa.Check.ge(rules["min"]))
        elif "max" in rules:
            checks.append(pa.Check.le(rules["max"]))
        columns[col_name] = pa.Column(
            checks=checks,
            nullable=nullable,
            unique=unique,
            required=True,
        )
    return pa.DataFrameSchema(columns=columns, coerce=False)


def validate(df, table_name, checks_file=None):
    """Run Pandera checks on a pandas DataFrame."""
    if checks_file is None:
        checks_file = CHECKS_DIR / f"{table_name}.yaml"

    if not checks_file.exists():
        log.warning("No checks file found: %s, skipping validation", checks_file)
        return QualityResult(table=table_name, success=True, rows=len(df))

    with open(checks_file) as f:
        spec = yaml.safe_load(f)

    failures = []
    total_checks = 0
    passed = 0

    # Check min_rows
    min_rows = spec.get("min_rows", 0)
    if min_rows:
        total_checks += 1
        if len(df) >= min_rows:
            passed += 1
        else:
            failures.append(f"row_count: {len(df)} < {min_rows}")

    # Check composite uniqueness
    for unique_cols in spec.get("unique", []):
        total_checks += 1
        dupes = df.duplicated(subset=unique_cols, keep=False).sum()
        if dupes == 0:
            passed += 1
        else:
            failures.append(f"duplicate_count({', '.join(unique_cols)}): {dupes} duplicates")

    # Build and run Pandera schema
    schema = _build_schema(spec, df.columns.tolist())
    if schema.columns:
        try:
            schema.validate(df, lazy=True)
            total_checks += len(schema.columns)
            passed += len(schema.columns)
        except pa.errors.SchemaErrors as e:
            failed_cols = set()
            for err in e.schema_errors:
                col = err.schema.name if hasattr(err, "schema") and err.schema else str(err)
                failed_cols.add(col)
            for col_name in schema.columns:
                total_checks += 1
                if col_name in failed_cols:
                    failures.append(f"column_check({col_name})")
                else:
                    passed += 1

    success = len(failures) == 0
    result = QualityResult(
        table=table_name,
        success=success,
        total_checks=total_checks,
        passed=passed,
        failed=len(failures),
        rows=len(df),
        failures=failures,
    )

    log.info(
        "Quality check [%s]: %s — %d/%d passed, %d failed, %d rows",
        table_name,
        "PASS" if success else "FAIL",
        passed,
        total_checks,
        len(failures),
        len(df),
    )
    if failures:
        for f in failures:
            log.error("  FAILED: %s", f)

    return result


def push_quality_metrics(result):
    """Push data quality metrics to VictoriaMetrics."""
    lines = [
        f'data_quality_passed{{table="{result.table}"}} {1 if result.success else 0}',
        f'data_quality_checks_total{{table="{result.table}"}} {result.total_checks}',
        f'data_quality_checks_passed{{table="{result.table}"}} {result.passed}',
        f'data_quality_checks_failed{{table="{result.table}"}} {result.failed}',
        f'data_quality_rows_total{{table="{result.table}"}} {result.rows}',
    ]

    body = "\n".join(lines) + "\n"
    try:
        resp = requests.post(
            f"{VM_URL}/api/v1/import/prometheus",
            data=body,
            headers={"Content-Type": "text/plain"},
            timeout=10,
        )
        resp.raise_for_status()
        log.info("Quality metrics pushed for table=%s", result.table)
    except Exception as e:
        log.warning("Failed to push quality metrics: %s", e)


def validate_and_push(df, table_name, checks_file=None, fail_on_error=True):
    """Validate + push metrics + optionally exit on failure."""
    result = validate(df, table_name, checks_file)
    push_quality_metrics(result)

    if not result.success and fail_on_error:
        log.error("Data quality check FAILED for %s, aborting", table_name)
        sys.exit(1)

    return result
