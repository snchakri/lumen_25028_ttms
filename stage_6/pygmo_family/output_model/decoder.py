
"""output_model.decoder
Responsible for transforming PyGMO optimisation results (vectors / individuals)
back into domain-specific schedule representations (pandas.DataFrame).

This module strictly adheres to Stage-6.4 foundational design and mathematical
bijection guarantees defined in Representation Layer. All conversions are
provably loss-less and validated in ‘output_model.validator’.
"""
from __future__ import annotations

import math
import pandas as pd
from typing import Dict, List, Tuple, Any
from pathlib import Path

from ..processing.representation import pygmo_to_course_dict  # bijective converter
from ..config import MEMORY_CONFIG
from ..input_model.context import InputModelContext
from ..processing.validation import ProcessingResult

__all__ = [
    'ScheduleDecoder',
]

class ScheduleDecoder:
    """Decode PyGMO optimisation results into tabular schedules.

    Attributes
    ----------
    context : InputModelContext
        Immutable context carrying bijection mappings & meta-data.
    """

    __slots__ = ("_ctx",)

    def __init__(self, context: InputModelContext) -> None:
        self._ctx = context  # reference only, do **not** copy large data

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def decode_individual(
        self, *, vector: List[float], course_order: List[str]
    ) -> pd.DataFrame:
        """Decode a single PyGMO vector into a pandas.DataFrame schedule.

        Parameters
        ----------
        vector : List[float]
            Normalised decision vector produced by PyGMO.
        course_order : List[str]
            Canonical ordering of courses used during encoding.

        Returns
        -------
        pd.DataFrame
            Tidy schedule with columns [course, faculty, room, timeslot, batch].
        """
        assignment_dict = pygmo_to_course_dict(
            vector=vector,
            course_order=course_order,
            max_values=self._ctx.bijection_data['max_values'],
        )

        # Build DataFrame
        df = (
            pd.DataFrame.from_dict(assignment_dict, orient='index', columns=['faculty', 'room', 'timeslot', 'batch'])
            .rename_axis('course')
            .reset_index()
        )
        return df

    def decode_pareto_front(
        self,
        *,
        pareto_front: List[List[float]],
        course_order: List[str],
    ) -> List[pd.DataFrame]:
        """Decode an entire Pareto front into a list of DataFrames."""
        return [self.decode_individual(vector=v, course_order=course_order) for v in pareto_front]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def export_schedule(df: pd.DataFrame, path: Path) -> None:
        """Export schedule to CSV with strict dtypes & deterministic order."""
        path.parent.mkdir(parents=True, exist_ok=True)
        df = df.astype({
            'course': 'string',
            'faculty': 'int32',
            'room': 'int32',
            'timeslot': 'int32',
            'batch': 'int32',
        })
        df.sort_values(['timeslot', 'room', 'course']).to_csv(path, index=False)
