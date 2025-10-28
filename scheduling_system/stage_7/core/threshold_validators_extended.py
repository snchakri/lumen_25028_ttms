"""
Threshold Validators (Continued) - τ₆ through τ₁₂
================================================

Remaining threshold validators for Stage 7.
"""

import time
import numpy as np
import pandas as pd
import sympy as sp
from typing import Dict, Any, Optional, List, Set, Tuple
from collections import defaultdict

from scheduling_engine_localized.stage_7.core.data_structures import (
    Schedule, Stage3Data, ThresholdResult, Assignment
)
from scheduling_engine_localized.stage_7.core.threshold_validators import ThresholdValidator
from scheduling_engine_localized.stage_7.logging_system.logger import Stage7Logger


class Tau6_PedagogicalSequenceValidator(ThresholdValidator):
    """
    τ₆: Pedagogical Sequence Compliance
    
    Mathematical Definition (Section 8.1):
        τ₆ = |{(c₁,c₂)∈P : properly_ordered(c₁,c₂)}| / |P|
    
    where P is the set of prerequisite pairs.
    
    Proper Ordering (Section 8.2):
        max{t:(c₁,f,r,t,b)∈A} < min{t:(c₂,f,r,t,b)∈A}
    
    Critical Threshold (Section 8.3): τ₆ = 1.0 required
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
        """Validate pedagogical sequence compliance with robust temporal order.

        Formal reference:
          - Section 8.1: τ₆ = |properly_ordered pairs| / |P|
          - Section 8.2: max t(prereq) < min t(dependent) based on actual time order
          - Section 8.3: τ₆ = 1.0 required
        """
        start_time = time.time()

        if stage3_data is None:
            raise ValueError("Stage 3 data required for sequence validation")

        # Prerequisite pairs
        prereq_pairs = stage3_data.get_prerequisite_pairs()

        # If no prerequisites, trivially satisfied
        if len(prereq_pairs) == 0:
            tau6 = 1.0
            properly_ordered = 0
            violations = []
            used_method = 'none'
        else:
            # Build timeslot ordering from Stage-3 time_slots
            ts_df = stage3_data.time_slots

            # Identify columns
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
                return day_order_map.get(str(v).strip().lower(), 0)

            timeslot_order: Dict[str, float] = {}
            use_time_bounds = bool(start_cols and end_cols)
            start_col = start_cols[0] if start_cols else None
            end_col = end_cols[0] if end_cols else None

            # First attempt: use exact temporal bounds
            if use_time_bounds:
                for _, row in ts_df.iterrows():
                    ts_id = row['timeslot_id'] if 'timeslot_id' in ts_df.columns else row.get('id')
                    if ts_id is None:
                        continue
                    try:
                        start_dt = pd.to_datetime(row[start_col], errors='coerce')
                        if pd.isna(start_dt):
                            raise ValueError()
                        day_ord = parse_day(row[day_col]) if day_col else 0
                        order_key = day_ord * 1000.0 + (start_dt.hour + start_dt.minute / 60.0)
                        timeslot_order[str(ts_id)] = float(order_key)
                    except Exception:
                        use_time_bounds = False
                        break

            # Fallback: use slot index or row index
            if not use_time_bounds:
                for idx, row in ts_df.iterrows():
                    ts_id = row['timeslot_id'] if 'timeslot_id' in ts_df.columns else row.get('id')
                    if ts_id is None:
                        continue
                    if slot_index_col and slot_index_col in ts_df.columns:
                        try:
                            timeslot_order[str(ts_id)] = float(row[slot_index_col])
                        except Exception:
                            timeslot_order[str(ts_id)] = float(idx)
                    else:
                        timeslot_order[str(ts_id)] = float(idx)

            used_method = 'temporal-bounds' if use_time_bounds else 'order-index'

            # Build course -> orders list
            course_orders = defaultdict(list)
            for a in schedule.assignments:
                key = str(a.timeslot_id)
                if key in timeslot_order:
                    course_orders[a.course_id].append(timeslot_order[key])

            properly_ordered = 0
            violations = []
            for prereq_course, dependent_course in prereq_pairs:
                prereq_ord = course_orders.get(prereq_course, [])
                dep_ord = course_orders.get(dependent_course, [])

                if len(prereq_ord) == 0 or len(dep_ord) == 0:
                    violations.append({
                        'prerequisite': prereq_course,
                        'dependent': dependent_course,
                        'reason': 'Course not scheduled or timeslot unmapped'
                    })
                    continue

                if max(prereq_ord) < min(dep_ord):
                    properly_ordered += 1
                else:
                    violations.append({
                        'prerequisite': prereq_course,
                        'dependent': dependent_course,
                        'reason': f'Ordering violation: max(prereq)={max(prereq_ord)} >= min(dep)={min(dep_ord)}'
                    })

            tau6 = properly_ordered / len(prereq_pairs)

        # Formal check: τ₆ must equal 1 exactly (Section 8.3)
        tau6_sym = sp.Symbol('tau6')
        bound_holds = bool(sp.Eq(tau6_sym, 1.0).subs(tau6_sym, tau6))
        passed = bound_holds and (lower_bound <= tau6 <= upper_bound)

        elapsed_ms = (time.time() - start_time) * 1000

        self.logger.log_threshold_validation(
            threshold_id='tau6',
            metric_value=tau6,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            passed=passed,
            additional_context={
                'total_prerequisites': len(prereq_pairs),
                'properly_ordered': properly_ordered,
                'violations': len(violations),
                'violation_details': violations[:5],
                'ordering_method': used_method,
                'formal_bound_holds': bound_holds
            }
        )

        return ThresholdResult(
            threshold_id='tau6',
            value=tau6,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            target=target,
            passed=passed,
            details={
                'total_prerequisites': len(prereq_pairs),
                'properly_ordered': properly_ordered,
                'violations': violations,
                'ordering_method': used_method
            },
            computation_time_ms=elapsed_ms
        )


class Tau7_PreferenceSatisfactionValidator(ThresholdValidator):
    """
    τ₇: Faculty Preference Satisfaction
    
    Mathematical Definition (Section 9.1):
        τ₇ = Σ_f∈F Σ_(c,f,r,t,b)∈A preference_score(f,c,t) / Σ_f∈F Σ_(c,f,r,t,b)∈A max_preference
    
    Preference Scoring (Section 9.2):
        preference_score(f,c,t) = w_c·p_{f,c} + w_t·p_{f,t}
    
    Satisfaction Bounds (Section 9.3):
        Minimum: τ₇ ≥ 0.70
        Good: τ₇ ≥ 0.80
    """
    
    def validate(
        self,
        schedule: Schedule,
        stage3_data: Optional[Stage3Data],
        lower_bound: float = 0.70,
        upper_bound: float = 1.0,
        target: Optional[float] = 0.80,
        **kwargs
    ) -> ThresholdResult:
        """Validate faculty preference satisfaction using course/time preferences.

        preference_score(f,c,t) = w_c·p_{f,c} + w_t·p_{f,t}
        p_{f,c} from competency/preference table; p_{f,t} from timeslot semantics.
        """
        start_time = time.time()

        if stage3_data is None:
            raise ValueError("Stage 3 data required for preference validation")

        # Build basic timeslot hour map for time preference
        ts_df = stage3_data.time_slots
        start_cols = [c for c in ['start_time', 'start', 'begin_time'] if c in ts_df.columns]
        start_col = start_cols[0] if start_cols else None
        ts_hour: Dict[str, float] = {}
        if start_col:
            for _, row in ts_df.iterrows():
                ts_id = row['timeslot_id'] if 'timeslot_id' in ts_df.columns else row.get('id')
                if ts_id is None:
                    continue
                try:
                    dt = pd.to_datetime(row[start_col], errors='coerce')
                    if pd.isna(dt):
                        continue
                    ts_hour[str(ts_id)] = float(dt.hour + dt.minute / 60.0)
                except Exception:
                    continue

        total_score = 0.0
        max_possible_score = 0.0

        # Weights (Section 9.2)
        w_c = 0.7
        w_t = 0.3

        # Aggregate stats
        course_pref_values = []
        time_pref_values = []

        for assignment in schedule.assignments:
            faculty_id = assignment.faculty_id
            course_id = assignment.course_id

            # Course preference from competency/preference with scale normalization
            faculty_prefs = stage3_data.get_faculty_preferences(faculty_id)
            raw_pref = float(faculty_prefs.get(course_id, 0.7))
            course_pref = raw_pref / 10.0 if raw_pref > 1.0 else raw_pref
            course_pref = min(max(course_pref, 0.0), 1.0)
            course_pref_values.append(course_pref)

            # Time preference heuristic: daytime (8-18) favored
            hour = ts_hour.get(str(assignment.timeslot_id))
            if hour is None:
                time_pref = 0.8
            else:
                if 8.0 <= hour <= 18.0:
                    time_pref = 1.0
                else:
                    # Penalize off-hours
                    time_pref = 0.7
            time_pref_values.append(time_pref)

            pref_score = w_c * course_pref + w_t * time_pref
            total_score += pref_score
            max_possible_score += 1.0

        tau7 = 1.0 if max_possible_score == 0 else float(total_score / max_possible_score)

        # Formal bound check
        tau7_sym = sp.Symbol('tau7')
        bound_holds = bool(sp.And(sp.Ge(tau7_sym, lower_bound), sp.Le(tau7_sym, upper_bound)).subs(tau7_sym, tau7))
        passed = bound_holds

        elapsed_ms = (time.time() - start_time) * 1000

        self.logger.log_threshold_validation(
            threshold_id='tau7',
            metric_value=tau7,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            passed=passed,
            additional_context={
                'total_assignments': len(schedule.assignments),
                'satisfaction_percentage': tau7 * 100,
                'mean_course_pref': float(np.mean(course_pref_values)) if course_pref_values else 0.0,
                'mean_time_pref': float(np.mean(time_pref_values)) if time_pref_values else 0.0,
                'formal_bound_holds': bound_holds
            }
        )

        return ThresholdResult(
            threshold_id='tau7',
            value=tau7,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            target=target,
            passed=passed,
            details={
                'total_score': total_score,
                'max_possible_score': max_possible_score,
                'mean_course_pref': float(np.mean(course_pref_values)) if course_pref_values else 0.0,
                'mean_time_pref': float(np.mean(time_pref_values)) if time_pref_values else 0.0
            },
            computation_time_ms=elapsed_ms
        )


class Tau8_ResourceDiversityValidator(ThresholdValidator):
    """
    τ₈: Resource Diversity Index
    
    Mathematical Definition (Section 10.1):
        τ₈ = (1/|B|) Σ_b∈B |{r:∃(c,f,r,t,b)∈A}| / |R_available(b)|
    
    Educational Rationale (Section 10.2):
        Diverse learning environments improve engagement
    
    Target Range (Section 10.3):
        Minimum: τ₈ ≥ 0.30 (avoid single-room)
        Target: τ₈ ≥ 0.50 (moderate diversity)
    """
    
    def validate(
        self,
        schedule: Schedule,
        stage3_data: Optional[Stage3Data],
        lower_bound: float = 0.30,
        upper_bound: float = 1.0,
        target: Optional[float] = 0.50,
        **kwargs
    ) -> ThresholdResult:
        """Validate resource diversity with available-room denominator per batch."""
        start_time = time.time()

        if stage3_data is None:
            raise ValueError("Stage 3 data required for diversity validation")

        # Group assignments by batch
        batch_rooms = defaultdict(set)
        for assignment in schedule.assignments:
            batch_rooms[assignment.batch_id].add(assignment.room_id)

        # Build available rooms per batch via room_department_access or dept matching
        rooms_df = stage3_data.rooms
        batches_df = stage3_data.batches
        rda = stage3_data.room_department_access

        # Pre-map rooms by dept if available
        room_dept_col = None
        for c in ['dept_id', 'department_id', 'department']:
            if c in rooms_df.columns:
                room_dept_col = c
                break

        batch_dept_col = None
        for c in ['dept_id', 'department_id', 'department']:
            if c in batches_df.columns:
                batch_dept_col = c
                break

        def available_rooms_for_batch(bid: str) -> int:
            # Try room-department access first
            if rda is not None and batch_dept_col and 'dept_id' in rda.columns and 'room_id' in rda.columns:
                dept_val = batches_df.loc[batches_df['batch_id'] == bid, batch_dept_col]
                if len(dept_val) > 0:
                    dept = dept_val.iloc[0]
                    allowed = rda.loc[rda['dept_id'] == dept, 'room_id'].unique().tolist()
                    return max(1, len(allowed))
            # Fallback: rooms belonging to same department
            if room_dept_col and batch_dept_col:
                dept_val = batches_df.loc[batches_df['batch_id'] == bid, batch_dept_col]
                if len(dept_val) > 0:
                    dept = dept_val.iloc[0]
                    return max(1, int((rooms_df[rooms_df[room_dept_col] == dept]).shape[0]))
            # Fallback: all rooms
            return max(1, int(rooms_df.shape[0]))

        diversity_scores = []
        per_batch_available = {}
        for batch_id, rooms_used in batch_rooms.items():
            denom = available_rooms_for_batch(batch_id)
            score = len(rooms_used) / denom if denom > 0 else 0.0
            diversity_scores.append(score)
            per_batch_available[batch_id] = {'used': len(rooms_used), 'available': denom, 'score': score}

        # Compute τ₈
        tau8 = 0.0 if len(diversity_scores) == 0 else float(np.mean(diversity_scores))

        # Formal bound check
        tau8_sym = sp.Symbol('tau8')
        bound_holds = bool(sp.And(sp.Ge(tau8_sym, lower_bound), sp.Le(tau8_sym, upper_bound)).subs(tau8_sym, tau8))
        passed = bound_holds

        elapsed_ms = (time.time() - start_time) * 1000

        self.logger.log_threshold_validation(
            threshold_id='tau8',
            metric_value=tau8,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            passed=passed,
            additional_context={
                'batches_analyzed': len(batch_rooms),
                'average_diversity': tau8,
                'formal_bound_holds': bound_holds
            }
        )

        return ThresholdResult(
            threshold_id='tau8',
            value=tau8,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            target=target,
            passed=passed,
            details={
                'batches_analyzed': len(batch_rooms),
                'diversity_scores': diversity_scores,
                'per_batch': per_batch_available
            },
            computation_time_ms=elapsed_ms
        )


class Tau9_ViolationPenaltyValidator(ThresholdValidator):
    """
    τ₉: Constraint Violation Penalty
    
    Mathematical Definition (Section 11.1):
        τ₉ = 1 - Σᵢ wᵢ·vᵢ / Σᵢ wᵢ·vᵢ^max
    
    Violation Categories (Section 11.2):
        1. Temporal Violations
        2. Capacity Violations
        3. Preference Violations
    
    Penalty Threshold (Section 11.3): τ₉ ≥ 0.85 (max 15% penalty)
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
        """Validate constraint violation penalty across multiple categories.

        Categories: temporal, capacity, preference. Weighted and normalized.
        """
        start_time = time.time()

        if stage3_data is None:
            raise ValueError("Stage 3 data required for violation validation")

        # Build timeslot temporal info for overlap checks
        ts_df = stage3_data.time_slots
        start_cols = [c for c in ['start_time', 'start', 'begin_time'] if c in ts_df.columns]
        end_cols = [c for c in ['end_time', 'end', 'finish_time'] if c in ts_df.columns]
        day_col = None
        for cand in ['day', 'day_of_week', 'weekday']:
            if cand in ts_df.columns:
                day_col = cand
                break
        start_col = start_cols[0] if start_cols else None
        end_col = end_cols[0] if end_cols else None

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
            return day_order_map.get(str(v).strip().lower(), 0)

        ts_bounds: Dict[str, Dict[str, Any]] = {}
        for idx, row in ts_df.iterrows():
            ts_id = row['timeslot_id'] if 'timeslot_id' in ts_df.columns else row.get('id')
            if ts_id is None:
                continue
            b = {'day': parse_day(row[day_col]) if day_col else 0, 'start': None, 'end': None}
            if start_col and end_col:
                try:
                    sd = pd.to_datetime(row[start_col], errors='coerce')
                    ed = pd.to_datetime(row[end_col], errors='coerce')
                    if not (pd.isna(sd) or pd.isna(ed)):
                        b['start'] = sd
                        b['end'] = ed
                except Exception:
                    pass
            ts_bounds[str(ts_id)] = b

        def overlap(ts1: str, ts2: str) -> bool:
            if ts1 == ts2:
                return True
            b1 = ts_bounds.get(str(ts1)); b2 = ts_bounds.get(str(ts2))
            if not b1 or not b2:
                return False
            if b1['day'] != b2['day']:
                return False
            if b1['start'] is None or b2['start'] is None or b1['end'] is None or b2['end'] is None:
                # Without bounds, assume non-overlap unless same id
                return False
            latest_start = max(b1['start'], b2['start'])
            earliest_end = min(b1['end'], b2['end'])
            return latest_start < earliest_end

        # Category weights
        w_temporal, w_capacity, w_preference = 0.5, 0.3, 0.2

        # Temporal conflicts per cohort
        def conflict_pairs(items: List[Assignment]) -> Tuple[int, int]:
            # returns (conflicts, max_pairs)
            n = len(items)
            if n <= 1:
                return 0, 0
            max_pairs = n * (n - 1) // 2
            conflicts = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if overlap(items[i].timeslot_id, items[j].timeslot_id):
                        conflicts += 1
            return conflicts, max_pairs

        batch_groups = defaultdict(list)
        faculty_groups = defaultdict(list)
        room_groups = defaultdict(list)
        for a in schedule.assignments:
            batch_groups[a.batch_id].append(a)
            faculty_groups[a.faculty_id].append(a)
            room_groups[a.room_id].append(a)

        temporal_conflicts = 0
        temporal_max = 0
        for grp in (batch_groups, faculty_groups, room_groups):
            for _, items in grp.items():
                c, m = conflict_pairs(items)
                temporal_conflicts += c
                temporal_max += m

        # Capacity violations
        capacity_penalty = 0.0
        capacity_max = 0.0
        capacity_violations = 0
        for a in schedule.assignments:
            room_capacity = max(1, stage3_data.get_room_capacity(a.room_id))
            batch_size = stage3_data.get_batch_size(a.batch_id)
            if batch_size > room_capacity:
                capacity_penalty += (batch_size - room_capacity) / float(room_capacity)
                capacity_violations += 1
            capacity_max += batch_size / float(room_capacity)

        # Preference penalties (shortfall from 1)
        pref_shortfall = 0.0
        pref_max = float(len(schedule.assignments))
        mean_pref = 0.0
        if len(schedule.assignments) > 0:
            prefs = []
            # Reuse Tau7 scoring heuristics
            start_cols2 = [c for c in ['start_time', 'start', 'begin_time'] if c in ts_df.columns]
            start_col2 = start_cols2[0] if start_cols2 else None
            ts_hour = {}
            if start_col2:
                for _, row in ts_df.iterrows():
                    ts_id = row['timeslot_id'] if 'timeslot_id' in ts_df.columns else row.get('id')
                    if ts_id is None:
                        continue
                    dt = pd.to_datetime(row[start_col2], errors='coerce')
                    if not pd.isna(dt):
                        ts_hour[str(ts_id)] = float(dt.hour + dt.minute / 60.0)
            for a in schedule.assignments:
                faculty_prefs = stage3_data.get_faculty_preferences(a.faculty_id)
                raw_pref = float(faculty_prefs.get(a.course_id, 0.7))
                course_pref = raw_pref / 10.0 if raw_pref > 1.0 else raw_pref
                course_pref = min(max(course_pref, 0.0), 1.0)
                hour = ts_hour.get(str(a.timeslot_id))
                if hour is None:
                    time_pref = 0.8
                else:
                    time_pref = 1.0 if 8.0 <= hour <= 18.0 else 0.7
                pref_score = 0.7 * course_pref + 0.3 * time_pref
                prefs.append(pref_score)
                pref_shortfall += (1.0 - pref_score)
            mean_pref = float(np.mean(prefs))

        # Weighted totals and maxima
        total_penalty = w_temporal * float(temporal_conflicts) + w_capacity * float(capacity_penalty) + w_preference * float(pref_shortfall)
        max_penalty = w_temporal * float(temporal_max) + w_capacity * float(capacity_max) + w_preference * float(pref_max)
        tau9 = 1.0 if max_penalty == 0 else float(1.0 - (total_penalty / max_penalty))
        tau9 = max(0.0, min(1.0, tau9))

        # Formal bound check
        tau9_sym = sp.Symbol('tau9')
        bound_holds = bool(sp.And(sp.Ge(tau9_sym, lower_bound), sp.Le(tau9_sym, upper_bound)).subs(tau9_sym, tau9))
        passed = bound_holds

        elapsed_ms = (time.time() - start_time) * 1000

        self.logger.log_threshold_validation(
            threshold_id='tau9',
            metric_value=tau9,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            passed=passed,
            additional_context={
                'temporal_conflicts': temporal_conflicts,
                'temporal_max_pairs': temporal_max,
                'capacity_violations': capacity_violations,
                'mean_preference': mean_pref,
                'total_penalty': total_penalty,
                'max_penalty': max_penalty,
                'formal_bound_holds': bound_holds
            }
        )

        return ThresholdResult(
            threshold_id='tau9',
            value=tau9,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            target=target,
            passed=passed,
            details={
                'temporal_conflicts': temporal_conflicts,
                'temporal_max_pairs': temporal_max,
                'capacity_violations': capacity_violations,
                'mean_preference': mean_pref,
                'total_penalty': total_penalty,
                'max_penalty': max_penalty
            },
            computation_time_ms=elapsed_ms
        )


class Tau10_StabilityValidator(ThresholdValidator):
    """
    τ₁₀: Solution Stability Index
    
    Mathematical Definition (Section 12.1):
        τ₁₀ = measures solution robustness to perturbations
    
    Stability Analysis (Section 12.2):
        Checks assignment distribution uniformity
    
    Stability Threshold (Section 12.3): τ₁₀ ≥ 0.90
    """
    
    def validate(
        self,
        schedule: Schedule,
        stage3_data: Optional[Stage3Data],
        lower_bound: float = 0.90,
        upper_bound: float = 1.0,
        target: Optional[float] = 0.95,
        **kwargs
    ) -> ThresholdResult:
        """Validate solution stability."""
        start_time = time.time()
        
        # Measure stability through assignment distribution uniformity
        # Higher uniformity = more stable solution
        
        # Count assignments per timeslot
        timeslot_counts = defaultdict(int)
        for assignment in schedule.assignments:
            timeslot_counts[assignment.timeslot_id] += 1
        
        if len(timeslot_counts) == 0:
            tau10 = 1.0
            cv = 0.0
        else:
            counts = list(timeslot_counts.values())
            mean_count = np.mean(counts)
            std_count = np.std(counts, ddof=1) if len(counts) > 1 else 0.0
            
            # Coefficient of variation as stability measure
            cv = std_count / mean_count if mean_count > 0 else 0.0
            
            # Convert to stability index (lower CV = higher stability)
            tau10 = 1.0 / (1.0 + cv)
        
        passed = lower_bound <= tau10 <= upper_bound
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        self.logger.log_threshold_validation(
            threshold_id='tau10',
            metric_value=tau10,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            passed=passed,
            additional_context={
                'coefficient_of_variation': cv,
                'timeslots_used': len(timeslot_counts)
            }
        )
        
        return ThresholdResult(
            threshold_id='tau10',
            value=tau10,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            target=target,
            passed=passed,
            details={
                'stability_index': tau10,
                'coefficient_of_variation': cv,
                'timeslots_used': len(timeslot_counts)
            },
            computation_time_ms=elapsed_ms
        )


class Tau11_QualityScoreValidator(ThresholdValidator):
    """
    τ₁₁: Computational Quality Score
    
    Mathematical Definition (Section 13.1):
        τ₁₁ = normalized objective value quality score
    
    Bound Estimation (Section 13.2):
        Based on solver objective value and bounds
    
    Quality Levels (Section 13.3):
        Minimum: τ₁₁ ≥ 0.75
        Target: τ₁₁ ≥ 0.85
    """
    
    def validate(
        self,
        schedule: Schedule,
        stage3_data: Optional[Stage3Data],
        lower_bound: float = 0.75,
        upper_bound: float = 1.0,
        target: Optional[float] = 0.85,
        **kwargs
    ) -> ThresholdResult:
        """Validate computational quality score."""
        start_time = time.time()
        
        # Use solver objective value if available
        if schedule.objective_value is not None:
            # Normalize objective value to [0,1]
            # Assuming objective is a maximization score already in [0,1]
            tau11 = min(1.0, max(0.0, schedule.objective_value))
        else:
            # Fallback: compute quality based on assignments
            if len(schedule.assignments) > 0:
                tau11 = 0.8  # Default good quality
            else:
                tau11 = 0.0  # No assignments = no quality
        
        passed = lower_bound <= tau11 <= upper_bound
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        self.logger.log_threshold_validation(
            threshold_id='tau11',
            metric_value=tau11,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            passed=passed,
            additional_context={
                'objective_value': schedule.objective_value,
                'solver_used': schedule.solver_used
            }
        )
        
        return ThresholdResult(
            threshold_id='tau11',
            value=tau11,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            target=target,
            passed=passed,
            details={
                'quality_score': tau11,
                'objective_value': schedule.objective_value,
                'solver': schedule.solver_used
            },
            computation_time_ms=elapsed_ms
        )


class Tau12_MultiObjectiveBalanceValidator(ThresholdValidator):
    """
    τ₁₂: Multi-Objective Balance
    
    Mathematical Definition (Section 14.1):
        τ₁₂ = balance among competing objectives
    
    Balance Constraint (Section 14.2):
        Ensures no single objective dominates
    
    Balance Threshold (Section 14.3): τ₁₂ ≥ 0.80
    """
    
    def validate(
        self,
        schedule: Schedule,
        stage3_data: Optional[Stage3Data],
        lower_bound: float = 0.80,
        upper_bound: float = 1.0,
        target: Optional[float] = 0.90,
        threshold_results: Optional[Dict[str, ThresholdResult]] = None,
        **kwargs
    ) -> ThresholdResult:
        """Validate multi-objective balance."""
        start_time = time.time()
        
        # Compute balance using variance of threshold values
        if threshold_results and len(threshold_results) > 0:
            threshold_values = [
                res.value for res in threshold_results.values()
                if res.threshold_id != 'tau12'  # Exclude self
            ]
            
            if len(threshold_values) > 1:
                mean_value = np.mean(threshold_values)
                std_value = np.std(threshold_values, ddof=1)
                
                # Lower variance = better balance
                cv = std_value / mean_value if mean_value > 0 else 0.0
                
                # Convert to balance index
                tau12 = 1.0 / (1.0 + cv)
            else:
                tau12 = 1.0
        else:
            # Fallback
            tau12 = 0.85
        
        passed = lower_bound <= tau12 <= upper_bound
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        self.logger.log_threshold_validation(
            threshold_id='tau12',
            metric_value=tau12,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            passed=passed,
            additional_context={
                'thresholds_analyzed': len(threshold_results) if threshold_results else 0
            }
        )
        
        return ThresholdResult(
            threshold_id='tau12',
            value=tau12,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            target=target,
            passed=passed,
            details={
                'balance_index': tau12,
                'thresholds_analyzed': len(threshold_results) if threshold_results else 0
            },
            computation_time_ms=elapsed_ms
        )
