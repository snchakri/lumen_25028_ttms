"""
Threshold Validators - All 12 Threshold Variables
=================================================

Implements rigorous validation for all 12 threshold variables per theoretical foundations:
- τ₁: Course Coverage Ratio (Section 3)
- τ₂: Conflict Resolution Rate (Section 4)
- τ₃: Faculty Workload Balance Index (Section 5)
- τ₄: Room Utilization Efficiency (Section 6)
- τ₅: Student Schedule Density (Section 7)
- τ₆: Pedagogical Sequence Compliance (Section 8)
- τ₇: Faculty Preference Satisfaction (Section 9)
- τ₈: Resource Diversity Index (Section 10)
- τ₉: Constraint Violation Penalty (Section 11)
- τ₁₀: Solution Stability Index (Section 12)
- τ₁₁: Computational Quality Score (Section 13)
- τ₁₂: Multi-Objective Balance (Section 14)

Each validator implements the exact mathematical definition from foundations.

Author: LUMEN Team [TEAM-ID: 93912]
"""

import time
import sympy as sp
from scipy import stats
import numpy as np
from typing import Dict, Any, Optional, List, Set, Tuple
from collections import defaultdict
import pandas as pd

from scheduling_engine_localized.stage_7.core.data_structures import (
    Schedule, Stage3Data, ThresholdResult, Assignment
)
from scheduling_engine_localized.stage_7.logging_system.logger import Stage7Logger


class ThresholdValidator:
    """Base class for threshold validators."""
    
    def __init__(self, logger: Stage7Logger):
        """Initialize validator."""
        self.logger = logger
    
    def validate(
        self,
        schedule: Schedule,
        stage3_data: Optional[Stage3Data],
        **kwargs
    ) -> ThresholdResult:
        """
        Validate threshold.
        
        Must be implemented by subclasses.
        """
        raise NotImplementedError


class Tau1_CourseCoverageValidator(ThresholdValidator):
    """
    τ₁: Course Coverage Ratio
    
    Mathematical Definition (Section 3.1):
        τ₁ = |{c ∈ C : ∃(c,f,r,t,b) ∈ A}| / |C|
    
    Theorem 3.1: For acceptable timetable, τ₁ ≥ 0.95
    """
    
    def validate(
        self,
        schedule: Schedule,
        stage3_data: Optional[Stage3Data],
        lower_bound: float = 0.95,
        upper_bound: float = 1.0,
        target: Optional[float] = 1.0,
        **kwargs
    ) -> ThresholdResult:
        """
        Validate course coverage ratio.
        
        Per Algorithm 3.2: Course Coverage Validation
        """
        start_time = time.time()
        
        # Get required courses C
        if stage3_data is None:
            raise ValueError("Stage 3 data required for course coverage validation")
        
        all_courses = stage3_data.get_required_courses()
        
        # Get covered courses (Algorithm 3.2, lines 1-4)
        covered_courses = schedule.get_courses_scheduled()
        
        # Compute τ₁ (Algorithm 3.2, line 5)
        if len(all_courses) == 0:
            tau1 = 1.0  # No courses to cover
        else:
            tau1 = len(covered_courses) / len(all_courses)

        # Formal theorem check (Theorem 3.1: τ₁ ≥ 0.95)
        tau1_sym = sp.Symbol('tau1')
        theorem_3_1 = sp.Ge(tau1_sym, lower_bound)
        theorem_holds = theorem_3_1.subs(tau1_sym, tau1)
        
        # Validation (Algorithm 3.2, lines 6-8)
        passed = bool(theorem_holds and (tau1 <= upper_bound))
        
        # Identify uncovered courses
        uncovered_courses = list(all_courses - covered_courses)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        self.logger.log_threshold_validation(
            threshold_id='tau1',
            metric_value=tau1,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            passed=passed,
            additional_context={
                'total_courses': len(all_courses),
                'covered_courses': len(covered_courses),
                'uncovered_count': len(uncovered_courses),
                'uncovered_courses': uncovered_courses[:10],
                'theorem_3_1_holds': bool(theorem_holds)
            }
        )
        
        return ThresholdResult(
            threshold_id='tau1',
            value=tau1,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            target=target,
            passed=passed,
            details={
                'total_courses': len(all_courses),
                'covered_courses': len(covered_courses),
                'coverage_percentage': tau1 * 100,
                'uncovered_courses': uncovered_courses
            },
            computation_time_ms=elapsed_ms
        )


class Tau2_ConflictResolutionValidator(ThresholdValidator):
    """
    τ₂: Conflict Resolution Rate
    
    Mathematical Definition (Section 4.1):
        τ₂ = 1 - |{(a₁,a₂) ∈ A×A : conflict(a₁,a₂)}| / |A|²
    
    Conflict Definition (Section 4.2):
        conflict(a₁,a₂) ≡ (t₁=t₂) ∧ ((f₁=f₂) ∨ (r₁=r₂) ∨ (b₁=b₂))
    
    Theorem 4.2: For valid timetable, τ₂ = 1.0 (zero conflicts) required
    """
    
    def validate(
        self,
        schedule: Schedule,
        stage3_data: Optional[Stage3Data],
        lower_bound: float = 1.0,
        upper_bound: float = 1.0,
        target: Optional[float] = 1.0,
        **kwargs
    ) -> ThresholdResult:
        """
        Validate conflict resolution rate.
        
        Per Algorithm 4.3: Conflict Detection
        """
        start_time = time.time()
        
        assignments = schedule.assignments
        conflict_count = 0
        conflicts = []
        
        # Algorithm 4.3, lines 1-6: Check all pairs
        for i, a1 in enumerate(assignments):
            for j, a2 in enumerate(assignments):
                if i >= j:  # Skip same assignment and duplicates
                    continue
                
                # Definition 4.1: Check conflict condition
                same_timeslot = (a1.timeslot_id == a2.timeslot_id)
                same_faculty = (a1.faculty_id == a2.faculty_id)
                same_room = (a1.room_id == a2.room_id)
                same_batch = (a1.batch_id == a2.batch_id)
                
                if same_timeslot and (same_faculty or same_room or same_batch):
                    conflict_count += 1
                    conflicts.append({
                        'assignment1': a1.to_tuple(),
                        'assignment2': a2.to_tuple(),
                        'conflict_type': (
                            'faculty' if same_faculty else
                            'room' if same_room else
                            'batch'
                        )
                    })
        
        # Compute τ₂ (Algorithm 4.3, line 7)
        n_assignments = len(assignments)
        if n_assignments == 0:
            tau2 = 1.0
        else:
            tau2 = 1.0 - (conflict_count / (n_assignments ** 2))

        # Formal theorem check (Theorem 4.2: τ₂ = 1.0)
        tau2_sym = sp.Symbol('tau2')
        theorem_4_2 = sp.Eq(tau2_sym, 1.0)
        theorem_holds = theorem_4_2.subs(tau2_sym, tau2)
        passed = bool(theorem_holds)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        self.logger.log_threshold_validation(
            threshold_id='tau2',
            metric_value=tau2,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            passed=passed,
            additional_context={
                'total_assignments': n_assignments,
                'conflicts_detected': conflict_count,
                'conflicts': conflicts[:5],
                'theorem_4_2_holds': bool(theorem_holds)
            }
        )
        
        return ThresholdResult(
            threshold_id='tau2',
            value=tau2,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            target=target,
            passed=passed,
            details={
                'total_assignments': n_assignments,
                'conflicts_detected': conflict_count,
                'conflicts': conflicts
            },
            computation_time_ms=elapsed_ms
        )


class Tau3_WorkloadBalanceValidator(ThresholdValidator):
    """
    τ₃: Faculty Workload Balance Index
    
    Mathematical Definition (Section 5.1):
        τ₃ = 1 - σ_W / μ_W
    
    where σ_W and μ_W are standard deviation and mean of faculty workloads.
    
    Workload Calculation (Section 5.2):
        W_f = Σ_{(c,f,r,t,b)∈A} h_c
    
    Proposition 5.2: τ₃ ≥ 0.85 required (CV ≤ 0.15)
    """
    
    def validate(
        self,
        schedule: Schedule,
        stage3_data: Optional[Stage3Data],
        lower_bound: float = 0.85,
        upper_bound: float = 1.0,
        target: Optional[float] = 0.95,
        **kwargs
    ) -> ThresholdResult:
        """Validate faculty workload balance."""
        start_time = time.time()
        
        if stage3_data is None:
            raise ValueError("Stage 3 data required for workload balance validation")
        
        # Calculate workload for each faculty (Section 5.2)
        faculty_workloads = defaultdict(float)
        
        for assignment in schedule.assignments:
            hours = stage3_data.get_course_hours(assignment.course_id)
            faculty_workloads[assignment.faculty_id] += hours
        
        # Get all faculty (including those with zero workload)
        all_faculty = set(stage3_data.faculty['faculty_id'].values)
        for faculty_id in all_faculty:
            if faculty_id not in faculty_workloads:
                faculty_workloads[faculty_id] = 0.0
        
        workloads = list(faculty_workloads.values())
        
        if len(workloads) == 0:
            tau3 = 1.0
            mean_workload = 0.0
            std_workload = 0.0
            cv = 0.0
        else:
            # Compute statistics
            mean_workload = np.mean(workloads)
            std_workload = np.std(workloads, ddof=1) if len(workloads) > 1 else 0.0
            # Shapiro-Wilk test for normality (robustness)
            shapiro_p = stats.shapiro(workloads)[1] if len(workloads) > 2 else None
            # Compute τ₃ (Section 5.1)
            if mean_workload == 0:
                tau3 = 1.0
                cv = 0.0
            else:
                cv = std_workload / mean_workload  # Coefficient of variation
                tau3 = 1.0 - cv
        
        # Formal proposition check (Proposition 5.2: τ₃ ≥ 0.85)
        tau3_sym = sp.Symbol('tau3')
        prop_5_2 = sp.Ge(tau3_sym, lower_bound)
        prop_holds = prop_5_2.subs(tau3_sym, tau3)
        passed = bool(prop_holds and (tau3 <= upper_bound))
        
        elapsed_ms = (time.time() - start_time) * 1000

        self.logger.log_threshold_validation(
            threshold_id='tau3',
            metric_value=tau3,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            passed=passed,
            additional_context={
                'mean_workload': mean_workload,
                'std_workload': std_workload,
                'coefficient_of_variation': cv,
                'faculty_count': len(faculty_workloads),
                'min_workload': min(workloads) if workloads else 0,
                'max_workload': max(workloads) if workloads else 0,
                'shapiro_p_value': shapiro_p,
                'proposition_5_2_holds': bool(prop_holds)
            }
        )

        return ThresholdResult(
            threshold_id='tau3',
            value=tau3,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            target=target,
            passed=passed,
            details={
                'mean_workload': mean_workload,
                'std_workload': std_workload,
                'coefficient_of_variation': cv,
                'faculty_count': len(faculty_workloads),
                'workload_distribution': {
                    'min': min(workloads) if workloads else 0,
                    'max': max(workloads) if workloads else 0,
                    'median': np.median(workloads) if workloads else 0
                }
            },
            computation_time_ms=elapsed_ms
        )


class Tau4_RoomUtilizationValidator(ThresholdValidator):
    """
    τ₄: Room Utilization Efficiency
    
    Mathematical Definition (Section 6.1):
        τ₄ = Σᵣ∈R Uᵣ·effective_capacity(r) / Σᵣ∈R max_hours·total_capacity(r)
    
    Effective Capacity (Section 6.2):
        effective_capacity(r,b) = min(cap_r, s_b + buffer)
    
    Quality Bounds (Section 6.4):
        Minimum: τ₄ ≥ 0.60
        Good: τ₄ ≥ 0.75
    """
    
    def validate(
        self,
        schedule: Schedule,
        stage3_data: Optional[Stage3Data],
        lower_bound: float = 0.60,
        upper_bound: float = 0.95,
        target: Optional[float] = 0.75,
        **kwargs
    ) -> ThresholdResult:
        """Validate room utilization efficiency."""
        start_time = time.time()
        
        if stage3_data is None:
            raise ValueError("Stage 3 data required for room utilization validation")
        
        # Calculate room utilization
        # Track hours per room (each assignment = 1 timeslot occurrence)
        room_hours_used = defaultdict(float)  # Count of timeslot hours used
        room_student_hours = defaultdict(float)  # Student-hours (hours × students)
        
        for assignment in schedule.assignments:
            room_id = assignment.room_id
            batch_size = stage3_data.get_batch_size(assignment.batch_id)
            room_capacity = stage3_data.get_room_capacity(room_id)
            
            # Each assignment is 1 timeslot occurrence (typically 1 hour)
            # Use assignment duration if available, else assume 1 hour
            if assignment.duration:
                hours = assignment.duration / 60.0  # Convert minutes to hours
            else:
                hours = 1.0
            
            room_hours_used[room_id] += hours
            # Effective utilization: hours weighted by occupancy rate
            occupancy_rate = min(batch_size / room_capacity, 1.0) if room_capacity > 0 else 0.0
            room_student_hours[room_id] += hours * occupancy_rate * room_capacity
        
        # Get all rooms
        all_rooms = stage3_data.rooms
        total_capacity = 0.0
        utilized_capacity = 0.0

        # Compute weekly availability from time_slots (sum of durations across the week)
        ts_df = stage3_data.time_slots
        start_cols = [c for c in ['start_time', 'start', 'begin_time'] if c in ts_df.columns]
        end_cols = [c for c in ['end_time', 'end', 'finish_time'] if c in ts_df.columns]
        start_col = start_cols[0] if start_cols else None
        end_col = end_cols[0] if end_cols else None
        if start_col and end_col:
            try:
                durations_h = []
                for _, row in ts_df.iterrows():
                    sd = pd.to_datetime(row[start_col], errors='coerce')
                    ed = pd.to_datetime(row[end_col], errors='coerce')
                    if pd.isna(sd) or pd.isna(ed):
                        continue
                    d = max((ed - sd).total_seconds() / 3600.0, 0.0)
                    durations_h.append(d)
                max_hours_per_week = float(np.sum(durations_h)) if durations_h else 40.0
            except Exception:
                max_hours_per_week = 40.0
        else:
            # Fallback if no temporal bounds; use number of slots * median duration or 1h
            if 'duration_minutes' in ts_df.columns:
                try:
                    unit = float(np.median(ts_df['duration_minutes'].fillna(0))) / 60.0 or 1.0
                except Exception:
                    unit = 1.0
            elif 'duration' in ts_df.columns:
                try:
                    unit = float(np.median(ts_df['duration'].fillna(0))) / 60.0 or 1.0
                except Exception:
                    unit = 1.0
            else:
                unit = 1.0
            max_hours_per_week = float(len(ts_df)) * unit if len(ts_df) > 0 else 40.0

        for _, room_row in all_rooms.iterrows():
            room_id = room_row['room_id']
            capacity = room_row['capacity']

            total_capacity += max_hours_per_week * capacity

            if room_id in room_student_hours:
                utilized_capacity += room_student_hours[room_id]
        
        # Compute τ₄ (Section 6.1)
        if total_capacity == 0:
            tau4 = 0.0
        else:
            tau4 = utilized_capacity / total_capacity
        
        # Formal bound check (Section 6.4: τ₄ ≥ 0.60 for minimum, τ₄ ≥ 0.75 for good)
        tau4_sym = sp.Symbol('tau4')
        bound_check = sp.Ge(tau4_sym, lower_bound)
        bound_holds = bound_check.subs(tau4_sym, tau4)
        passed = bool(bound_holds and (tau4 <= upper_bound))

        elapsed_ms = (time.time() - start_time) * 1000

        # Statistical analysis: utilization distribution
        utilizations = [
            room_student_hours[r] / (max_hours_per_week * stage3_data.get_room_capacity(r))
            if stage3_data.get_room_capacity(r) > 0 else 0
            for r in room_student_hours
        ]
        mean_util = np.mean(utilizations) if utilizations else 0
        std_util = np.std(utilizations, ddof=1) if len(utilizations) > 1 else 0
        self.logger.log_threshold_validation(
            threshold_id='tau4',
            metric_value=tau4,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            passed=passed,
            additional_context={
                'total_rooms': len(all_rooms),
                'utilized_rooms': len(room_hours_used),
                'utilization_percentage': tau4 * 100,
                'mean_room_utilization': mean_util,
                'std_room_utilization': std_util,
                'formal_bound_holds': bool(bound_holds)
            }
        )

        return ThresholdResult(
            threshold_id='tau4',
            value=tau4,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            target=target,
            passed=passed,
            details={
                'total_rooms': len(all_rooms),
                'utilized_rooms': len(room_hours_used),
                'total_capacity': total_capacity,
                'utilized_capacity': utilized_capacity,
                'utilization_percentage': tau4 * 100
            },
            computation_time_ms=elapsed_ms
        )


class Tau5_ScheduleDensityValidator(ThresholdValidator):
    """
    τ₅: Student Schedule Density
    
    Mathematical Definition (Section 7.1):
        τ₅ = (1/|B|) Σ_b∈B scheduled_hours(b) / time_span(b)
    
    Time Span Calculation (Section 7.2):
        time_span(b) = max(T_b) - min(T_b) + 1
    
    where T_b = {t : ∃(c,f,r,t,b)∈A}
    """
    
    def validate(
        self,
        schedule: Schedule,
        stage3_data: Optional[Stage3Data],
        lower_bound: float = 0.60,
        upper_bound: float = 0.90,
        target: Optional[float] = 0.75,
        **kwargs
    ) -> ThresholdResult:
        """Validate student schedule density with robust time span semantics.

        Formal reference:
          - Section 7.1: τ₅ = (1/|B|) Σ_b scheduled_hours(b) / time_span(b)
          - Section 7.2: time_span(b) = max(T_b) - min(T_b) + 1 (timeslot units)
        """
        start_time = time.time()

        if stage3_data is None:
            raise ValueError("Stage 3 data required for schedule density validation")

        # Build timeslot metadata: order and duration
        ts_df = stage3_data.time_slots
        # Heuristics for columns
        day_col = None
        for cand in ['day', 'day_of_week', 'weekday']:
            if cand in ts_df.columns:
                day_col = cand
                break
        start_cols = [c for c in ['start_time', 'start', 'begin_time'] if c in ts_df.columns]
        end_cols = [c for c in ['end_time', 'end', 'finish_time'] if c in ts_df.columns]
        slot_index_col = None
        for cand in ['slot_index', 'timeslot_index', 'index', 'order']:
            if cand in ts_df.columns:
                slot_index_col = cand
                break

        # Map day ordering
        day_order_map = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }

        def parse_day(v) -> int:
            if pd.isna(v):
                return 0
            if isinstance(v, (int, float)):
                iv = int(v)
                return iv if 0 <= iv <= 6 else 0
            s = str(v).strip().lower()
            return day_order_map.get(s, 0)

        # Attempt to compute canonical order and duration per timeslot
        timeslot_meta: Dict[str, Dict[str, float]] = {}
        durations = []
        use_time_bounds = bool(start_cols and end_cols)
        start_col = start_cols[0] if start_cols else None
        end_col = end_cols[0] if end_cols else None

        for _, row in ts_df.iterrows():
            ts_id = row['timeslot_id'] if 'timeslot_id' in ts_df.columns else row.get('id')
            if ts_id is None:
                continue

            if use_time_bounds:
                try:
                    start_dt = pd.to_datetime(row[start_col], errors='coerce')
                    end_dt = pd.to_datetime(row[end_col], errors='coerce')
                    if pd.isna(start_dt) or pd.isna(end_dt):
                        raise ValueError()
                    dur_h = max((end_dt - start_dt).total_seconds() / 3600.0, 0.0)
                    # Order: day order then time within day if day exists
                    day_ord = parse_day(row[day_col]) if day_col else 0
                    order_key = day_ord * 1000.0 + (start_dt.hour + start_dt.minute / 60.0)
                    timeslot_meta[str(ts_id)] = {
                        'order': float(order_key),
                        'duration_h': float(dur_h),
                        'start_ts': start_dt,
                        'end_ts': end_dt,
                        'day_ord': float(day_ord),
                    }
                    if dur_h > 0:
                        durations.append(dur_h)
                except Exception:
                    use_time_bounds = False  # fallback to index-based
                    # Will fill in later in the next loop
            else:
                # Fill later using slot index or default 1 hour
                pass

        if not use_time_bounds:
            # Fallback: use slot index and assume uniform duration
            default_unit = 1.0
            if 'duration_minutes' in ts_df.columns:
                try:
                    default_unit = float(np.median(ts_df['duration_minutes'].fillna(0))) / 60.0 or 1.0
                except Exception:
                    default_unit = 1.0
            elif 'duration' in ts_df.columns:
                try:
                    default_unit = float(np.median(ts_df['duration'].fillna(0))) / 60.0 or 1.0
                except Exception:
                    default_unit = 1.0
            for idx, row in ts_df.iterrows():
                ts_id = row['timeslot_id'] if 'timeslot_id' in ts_df.columns else row.get('id')
                if ts_id is None:
                    continue
                order_val = float(row[slot_index_col]) if slot_index_col else float(idx)
                timeslot_meta[str(ts_id)] = {
                    'order': order_val,
                    'duration_h': default_unit,
                    'start_ts': None,
                    'end_ts': None,
                    'day_ord': float(parse_day(row[day_col])) if day_col else 0.0,
                }
                durations.append(timeslot_meta[str(ts_id)]['duration_h'])

        # Helper to get duration per assignment
        def assignment_duration_hours(a: Assignment) -> float:
            if a.duration is not None:
                try:
                    return max(float(a.duration) / 60.0, 0.0)
                except Exception:
                    pass
            # Use timeslot duration if available
            meta = timeslot_meta.get(str(a.timeslot_id))
            if meta:
                return float(meta.get('duration_h', 1.0))
            # Fallback to course hours divided by occurrences of that course in batch (rough)
            try:
                ch = stage3_data.get_course_hours(a.course_id)
                if ch and ch > 0:
                    return float(ch)
            except Exception:
                pass
            return 1.0

        # Group assignments by batch
        batch_assignments = defaultdict(list)
        for assignment in schedule.assignments:
            batch_assignments[assignment.batch_id].append(assignment)

        batch_densities = []
        per_batch_details = {}

        for batch_id, assignments in batch_assignments.items():
            if not assignments:
                continue
            # Collect orders and temporal bounds
            orders = []
            start_bounds = []
            end_bounds = []
            for a in assignments:
                meta = timeslot_meta.get(str(a.timeslot_id))
                if meta is not None:
                    orders.append(meta['order'])
                    if meta['start_ts'] is not None and meta['end_ts'] is not None:
                        start_bounds.append(meta['start_ts'])
                        end_bounds.append(meta['end_ts'])

            # Compute scheduled hours
            scheduled_hours = float(sum(assignment_duration_hours(a) for a in assignments))

            # Compute time span
            time_span_h = 0.0
            calc_method = 'index+unit'
            if start_bounds and end_bounds:
                earliest = min(start_bounds)
                latest = max(end_bounds)
                time_span_h = max((latest - earliest).total_seconds() / 3600.0, 0.0)
                calc_method = 'temporal-bounds'
            elif orders:
                unit = float(np.median(durations)) if durations else 1.0
                time_span_units = (max(orders) - min(orders) + 1.0)
                time_span_h = max(time_span_units * unit, 0.0)
                calc_method = 'order-span'

            density = 0.0
            if time_span_h > 0:
                density = float(scheduled_hours / time_span_h)
            # Clip to [0, 1.5] buffer then to 1.0 for final metric
            density_capped = float(min(max(density, 0.0), 1.5))
            batch_densities.append(min(density_capped, 1.0))
            per_batch_details[batch_id] = {
                'scheduled_hours': scheduled_hours,
                'time_span_hours': time_span_h,
                'raw_density': density,
                'density_capped': min(density_capped, 1.0),
                'calc_method': calc_method,
                'assignments': len(assignments)
            }

        # Compute τ₅ (Section 7.1)
        if len(batch_densities) == 0:
            tau5 = 0.0
        else:
            tau5 = float(np.mean(batch_densities))

        # Formal bound check using sympy
        tau5_sym = sp.Symbol('tau5')
        bound_check = sp.And(sp.Ge(tau5_sym, lower_bound), sp.Le(tau5_sym, upper_bound))
        bound_holds = bool(bound_check.subs(tau5_sym, tau5))
        passed = bound_holds

        elapsed_ms = (time.time() - start_time) * 1000

        # Stats
        mu = float(np.mean(batch_densities)) if batch_densities else 0.0
        sd = float(np.std(batch_densities, ddof=1)) if len(batch_densities) > 1 else 0.0
        mn = float(np.min(batch_densities)) if batch_densities else 0.0
        mx = float(np.max(batch_densities)) if batch_densities else 0.0

        self.logger.log_threshold_validation(
            threshold_id='tau5',
            metric_value=tau5,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            passed=passed,
            additional_context={
                'batches_analyzed': len(batch_densities),
                'mean_density': mu,
                'std_density': sd,
                'min_density': mn,
                'max_density': mx,
                'formal_bound_holds': bound_holds
            }
        )

        return ThresholdResult(
            threshold_id='tau5',
            value=tau5,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            target=target,
            passed=passed,
            details={
                'batches_analyzed': len(batch_densities),
                'density_distribution': {
                    'mean': mu,
                    'std': sd,
                    'min': mn,
                    'max': mx
                },
                'per_batch': per_batch_details
            },
            computation_time_ms=elapsed_ms
        )


# Continue in next file...
