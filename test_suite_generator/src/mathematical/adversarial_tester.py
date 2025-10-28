"""
Adversarial Testing Framework

Injects targeted violations and mutations to test detection coverage
of the validation framework.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import random
import logging

logger = logging.getLogger(__name__)


@dataclass
class AdversarialConfig:
    percentage: float = 0.05  # 5% entities mutated
    seed: Optional[int] = None


class AdversarialTester:
    """
    Adversarial testing framework for Type II test case generation.
    
    Intentionally injects constraint violations to verify that the
    validation framework detects them. Target: 100% detection rate.
    """
    
    def __init__(self, config: Optional[AdversarialConfig] = None):
        self.config = config or AdversarialConfig()
        if self.config.seed is not None:
            random.seed(self.config.seed)
        self.violations_injected = 0
        self.violations_detected = 0
        logger.info(
            f"AdversarialTester initialized (percentage={self.config.percentage})"
        )

    def inject_credit_limit_violations(
        self,
        students: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Inject credit limit violations into student data.
        
        Violations:
        - Credits > 27 (absolute limit)
        - Credits < 3 (under-enrolled)
        """
        mutated = []
        for s in students:
            if random.random() < self.config.percentage:
                s = dict(s)
                # Randomly choose over or under enrolled
                if random.random() < 0.7:
                    # Over-enrolled (more common violation)
                    s['total_credits'] = random.randint(28, 35)
                else:
                    # Under-enrolled
                    s['total_credits'] = random.randint(0, 2)
                s['adversarial'] = True
                s['adversarial_type'] = 'credit_limit_violation'
                self.violations_injected += 1
            mutated.append(s)
        return mutated

    def inject_prereq_cycle(
        self,
        edges: List[tuple]
    ) -> List[tuple]:
        """
        Inject prerequisite cycles into prerequisite graph.
        
        Creates cycles by adding reverse edges.
        """
        mutated = list(edges)
        if mutated and random.random() < self.config.percentage:
            # Create cycle from random edge
            idx = random.randint(0, len(mutated) - 1)
            a, b = mutated[idx]
            mutated.append((b, a))  # create cycle
            self.violations_injected += 1
            logger.debug(f"Injected prerequisite cycle: {b} -> {a}")
        return mutated

    def inject_room_capacity_violation(
        self,
        classes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Inject room capacity violations.
        
        Violations:
        - Enrollment > room capacity + 5% buffer
        - Negative capacity
        """
        mutated = []
        for c in classes:
            if random.random() < self.config.percentage:
                c = dict(c)
                cap = int(c.get('room_capacity', 30))
                if random.random() < 0.8:
                    # Overcapacity
                    buffer = int(cap * 1.05)
                    c['enrollment_count'] = buffer + random.randint(5, 20)
                else:
                    # Negative capacity
                    c['room_capacity'] = -random.randint(1, 10)
                c['adversarial'] = True
                c['adversarial_type'] = 'capacity_violation'
                self.violations_injected += 1
            mutated.append(c)
        return mutated
    
    def inject_foreign_key_violations(
        self,
        entities: List[Dict[str, Any]],
        fk_field: str
    ) -> List[Dict[str, Any]]:
        """
        Inject foreign key violations.
        
        Violations:
        - Reference to non-existent UUID
        - Invalid UUID format
        """
        mutated = []
        for e in entities:
            if random.random() < self.config.percentage:
                e = dict(e)
                if random.random() < 0.5:
                    # Non-existent UUID
                    import uuid
                    e[fk_field] = str(uuid.uuid4())
                else:
                    # Invalid UUID format
                    e[fk_field] = "invalid-uuid-format"
                e['adversarial'] = True
                e['adversarial_type'] = 'foreign_key_violation'
                self.violations_injected += 1
            mutated.append(e)
        return mutated
    
    def inject_temporal_violations(
        self,
        schedules: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Inject temporal violations.
        
        Violations:
        - Start time after end time
        - Overlapping time slots for same resource
        """
        mutated = []
        for s in schedules:
            if random.random() < self.config.percentage:
                s = dict(s)
                # Swap start and end times
                start = s.get('start_time')
                end = s.get('end_time')
                if start and end:
                    s['start_time'] = end
                    s['end_time'] = start
                s['adversarial'] = True
                s['adversarial_type'] = 'temporal_violation'
                self.violations_injected += 1
            mutated.append(s)
        return mutated
    
    def inject_data_type_violations(
        self,
        entities: List[Dict[str, Any]],
        field: str,
        expected_type: str
    ) -> List[Dict[str, Any]]:
        """
        Inject data type violations.
        
        Violations:
        - Wrong type for field
        - Out of range values
        - Invalid enum values
        """
        mutated = []
        for e in entities:
            if random.random() < self.config.percentage:
                e = dict(e)
                if expected_type == "positive_int":
                    e[field] = -random.randint(1, 100)
                elif expected_type == "date":
                    e[field] = "invalid-date-format"
                elif expected_type == "enum":
                    e[field] = "INVALID_ENUM_VALUE"
                else:
                    e[field] = None  # NULL violation
                e['adversarial'] = True
                e['adversarial_type'] = 'data_type_violation'
                self.violations_injected += 1
            mutated.append(e)
        return mutated
    
    def record_detection(self, detected_count: int) -> None:
        """Record number of violations detected by validation."""
        self.violations_detected += detected_count
    
    def mutation_score(self, detected: int, total: int) -> float:
        """Calculate mutation score (detection rate)."""
        return (detected / total * 100.0) if total > 0 else 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get adversarial testing statistics.
        
        Returns:
            Dictionary with violation injection and detection stats
        """
        detection_rate = self.mutation_score(
            self.violations_detected,
            self.violations_injected
        )
        
        return {
            "violations_injected": self.violations_injected,
            "violations_detected": self.violations_detected,
            "detection_rate": detection_rate,
            "injection_percentage": self.config.percentage * 100,
            "target_detection_rate": 100.0,
            "meets_target": detection_rate >= 100.0
        }
