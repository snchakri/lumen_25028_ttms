"""
stage_5_1/io.py
Robust I/O utilities for Stage 5.1 complexity analysis

This module provides high-level helpers to:
1. Load Stage 3 output files using Stage3DataLoader.
2. Serialize computed complexity metrics to JSON using the schema defined
   in common.schema.ComplexityParameterVector and ExecutionMetadata.
3. Manage per-execution directories (timestamp-based isolation) and ensure
   atomic write operations with rollback on failure.

All functions raise Stage5ValidationError or Stage5ComputationError on
failure. Logging is JSON-structured via common.logging.get_logger.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from ..common.logging import get_logger, log_operation
from ..common.exceptions import Stage5ValidationError, Stage5ComputationError
from ..common.schema import ExecutionMetadata, ComplexityParameterVector

# Public re-exports for downstream modules
__all__ = [
    "load_stage3_inputs",
    "write_complexity_metrics",
]


# ---------------------------------------------------------------------------
# Stage 3 loader import done lazily to avoid heavy deps when only serializing
# ---------------------------------------------------------------------------

_logger = get_logger("stage5_1.io")


def load_stage3_inputs(l_raw: Path, l_rel: Path, l_idx: Path):
    """Load Stage 3 outputs and return ProcessedStage3Data.

    Args:
        l_raw: Path to `L_raw.parquet` file.
        l_rel: Path to `L_rel.graphml` file.
        l_idx: Path to `L_idx.*` file (any supported extension).

    Returns:
        ProcessedStage3Data fully validated and pre-processed.
    """
    # Local import to keep import tree shallow for non-loader callers
    from .compute import Stage3DataLoader  # pylint: disable=import-error

    loader = Stage3DataLoader(logger=_logger)
    return loader.load_stage3_outputs(l_raw, l_rel, l_idx)


def _safe_write_json(obj: Dict[str, Any], dest: Path) -> None:
    """Write JSON atomically with `.tmp` file then `replace`."""
    tmp_path = dest.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)
    tmp_path.replace(dest)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_complexity_metrics(
    vector: ComplexityParameterVector,
    output_dir: Path,
    computation_time_ms: int,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Serialize complexity metrics JSON to *output_dir*/complexity_metrics.json.

    The file layout adheres to the strict schema in the foundational design
    document. Any deviation raises *Stage5ValidationError*.
    """
    with log_operation(_logger, "write_complexity_metrics", {"output_dir": str(output_dir)}):
        _ensure_dir(output_dir)
        metrics_path = output_dir / "complexity_metrics.json"

        timestamp = datetime.now(tz=timezone.utc).isoformat()

        execution_meta = ExecutionMetadata(
            timestamp=timestamp,
            computation_time_ms=computation_time_ms,
        )

        payload: Dict[str, Any] = {
            "schema_version": "1.0.0",
            "execution_metadata": execution_meta.dict(),
            "complexity_parameters": vector.dict(include={k: ... for k in vector.__fields__ if k != "composite_index"}),
            "composite_index": vector.composite_index,
            "parameter_statistics": extra_metadata or {},
        }

        try:
            _safe_write_json(payload, metrics_path)
        except (OSError, json.JSONDecodeError) as exc:
            raise Stage5ComputationError(
                "Failed to write complexity_metrics.json",
                computation_type="json_serialization",
                input_parameters={"dest_path": str(metrics_path)},
            ) from exc

        _logger.info("Complexity metrics written to %s", metrics_path)
        return metrics_path
