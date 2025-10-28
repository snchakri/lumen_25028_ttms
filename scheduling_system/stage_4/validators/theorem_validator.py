"""
Theorem Compliance Checker for Stage 4 Feasibility Check
Tracks theorem compliance across all layers and validates complexity bounds
"""

import logging
import time
import psutil
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ComplianceStatus(str, Enum):
    """Theorem compliance status"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class ComplexityBound:
    """Complexity bound specification"""
    theoretical: str  # e.g., "O(n log n)"
    measured: float  # Actual execution time or operation count
    data_size: int  # Input data size
    status: ComplianceStatus


@dataclass
class TheoremCompliance:
    """Theorem compliance tracking"""
    theorem: str
    layer: int
    status: ComplianceStatus
    conditions_checked: int
    conditions_passed: int
    complexity_bound: Optional[ComplexityBound]
    mathematical_invariants: Dict[str, bool]
    notes: List[str] = field(default_factory=list)


class TheoremComplianceChecker:
    """
    Tracks theorem compliance across all layers
    
    Features:
    - Validate complexity bounds (measure actual vs. theoretical)
    - Check mathematical invariants during execution
    - Generate compliance reports
    - Flag deviations from theoretical guarantees
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.compliance_records: List[TheoremCompliance] = []
        
        # Expected complexity bounds from foundations
        self.expected_complexity = {
            1: "O(n log n)",  # Layer 1: BCNF checking
            2: "O(|V| + |E|)",  # Layer 2: Cycle detection
            3: "O(N)",  # Layer 3: Resource capacity
            4: "O(n)",  # Layer 4: Temporal window
            5: "O(|C| × |F|)",  # Layer 5: Hall's theorem
            6: "O(n²)",  # Layer 6: Conflict graph
            7: "O(n²)"  # Layer 7: AC-3 propagation
        }
    
    def check_layer_compliance(
        self,
        layer_number: int,
        theorem: str,
        execution_time_ms: float,
        data_size: int,
        conditions_checked: int,
        conditions_passed: int,
        mathematical_invariants: Dict[str, bool]
    ) -> TheoremCompliance:
        """
        Check compliance for a specific layer
        
        Args:
            layer_number: Layer number (1-7)
            theorem: Theorem name
            execution_time_ms: Actual execution time
            data_size: Input data size
            conditions_checked: Number of conditions checked
            conditions_passed: Number of conditions passed
            mathematical_invariants: Mathematical invariants and their status
            
        Returns:
            TheoremCompliance record
        """
        # Determine compliance status
        if conditions_passed == conditions_checked:
            status = ComplianceStatus.COMPLIANT
        elif conditions_passed == 0:
            status = ComplianceStatus.NON_COMPLIANT
        else:
            status = ComplianceStatus.PARTIALLY_COMPLIANT
        
        # Check complexity bound
        complexity_bound = self._analyze_complexity(
            layer_number,
            execution_time_ms,
            data_size
        )
        
        # Check mathematical invariants
        all_invariants_met = all(mathematical_invariants.values())
        
        # Create compliance record
        compliance = TheoremCompliance(
            theorem=theorem,
            layer=layer_number,
            status=status,
            conditions_checked=conditions_checked,
            conditions_passed=conditions_passed,
            complexity_bound=complexity_bound,
            mathematical_invariants=mathematical_invariants
        )
        
        # Add notes
        if not all_invariants_met:
            compliance.notes.append("Some mathematical invariants violated")
        
        if complexity_bound and complexity_bound.status == ComplianceStatus.NON_COMPLIANT:
            compliance.notes.append(
                f"Complexity bound violated: {complexity_bound.theoretical} expected, "
                f"measured {complexity_bound.measured:.2f}"
            )
        
        self.compliance_records.append(compliance)
        return compliance
    
    def _analyze_complexity(
        self,
        layer_number: int,
        execution_time_ms: float,
        data_size: int
    ) -> ComplexityBound:
        """
        Analyze complexity bound compliance
        
        Args:
            layer_number: Layer number
            execution_time_ms: Execution time
            data_size: Data size
            
        Returns:
            ComplexityBound with analysis
        """
        theoretical = self.expected_complexity.get(layer_number, "Unknown")
        
        # Simple complexity analysis based on data size
        # For O(n log n), expect roughly n * log(n) scaling
        # For O(n²), expect roughly n² scaling
        # For O(n), expect roughly n scaling
        
        if theoretical == "O(n log n)":
            expected_ratio = data_size * (1 + data_size.bit_length())  # Approximate n log n
            actual_ratio = execution_time_ms
            # Allow some variance (within 10x)
            compliant = actual_ratio <= expected_ratio * 10
        elif theoretical == "O(n²)":
            expected_ratio = data_size ** 2
            actual_ratio = execution_time_ms
            compliant = actual_ratio <= expected_ratio * 10
        elif theoretical == "O(n)" or theoretical == "O(N)":
            expected_ratio = data_size
            actual_ratio = execution_time_ms
            compliant = actual_ratio <= expected_ratio * 100  # More lenient for linear
        elif theoretical == "O(|V| + |E|)":
            # Graph complexity - harder to validate without graph structure
            expected_ratio = data_size * 2  # Rough estimate
            actual_ratio = execution_time_ms
            compliant = actual_ratio <= expected_ratio * 100
        elif theoretical == "O(|C| × |F|)":
            # Bipartite matching - quadratic in worst case
            expected_ratio = data_size ** 2
            actual_ratio = execution_time_ms
            compliant = actual_ratio <= expected_ratio * 10
        else:
            # Unknown complexity - cannot validate
            return ComplexityBound(
                theoretical=theoretical,
                measured=execution_time_ms,
                data_size=data_size,
                status=ComplianceStatus.NOT_APPLICABLE
            )
        
        status = ComplianceStatus.COMPLIANT if compliant else ComplianceStatus.NON_COMPLIANT
        
        return ComplexityBound(
            theoretical=theoretical,
            measured=execution_time_ms,
            data_size=data_size,
            status=status
        )
    
    def check_mathematical_invariants(
        self,
        layer_number: int,
        invariants: Dict[str, Any]
    ) -> Dict[str, bool]:
        """
        Check mathematical invariants for a layer
        
        Args:
            layer_number: Layer number
            invariants: Dictionary of invariant names and values
            
        Returns:
            Dictionary of invariant status
        """
        results = {}
        
        for invariant_name, invariant_value in invariants.items():
            if layer_number == 1:
                # Layer 1 invariants
                if invariant_name == "null_keys_zero":
                    results[invariant_name] = invariant_value == 0
                elif invariant_name == "all_keys_unique":
                    results[invariant_name] = invariant_value == True
                elif invariant_name == "no_fd_violations":
                    results[invariant_name] = invariant_value == 0
            
            elif layer_number == 2:
                # Layer 2 invariants
                if invariant_name == "no_fk_cycles":
                    results[invariant_name] = invariant_value == True
                elif invariant_name == "cardinality_satisfied":
                    results[invariant_name] = invariant_value == True
            
            elif layer_number == 3:
                # Layer 3 invariants
                if invariant_name == "all_demands_leq_supplies":
                    results[invariant_name] = invariant_value == True
                elif invariant_name == "pigeonhole_principle":
                    results[invariant_name] = invariant_value == True
            
            elif layer_number == 4:
                # Layer 4 invariants
                if invariant_name == "all_entities_temporal_feasible":
                    results[invariant_name] = invariant_value == True
            
            elif layer_number == 5:
                # Layer 5 invariants
                if invariant_name == "halls_condition_satisfied":
                    results[invariant_name] = invariant_value == True
                elif invariant_name == "bipartite_matching_exists":
                    results[invariant_name] = invariant_value == True
            
            elif layer_number == 6:
                # Layer 6 invariants
                if invariant_name == "max_degree_leq_timeslots":
                    results[invariant_name] = invariant_value == True
                elif invariant_name == "no_large_cliques":
                    results[invariant_name] = invariant_value == True
            
            elif layer_number == 7:
                # Layer 7 invariants
                if invariant_name == "no_domain_wipeout":
                    results[invariant_name] = invariant_value == True
                elif invariant_name == "arc_consistency_achieved":
                    results[invariant_name] = invariant_value == True
        
        return results
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive compliance report
        
        Returns:
            Dictionary with compliance summary
        """
        if not self.compliance_records:
            return {"message": "No compliance records available"}
        
        # Calculate statistics
        total_layers = len(self.compliance_records)
        compliant_layers = sum(
            1 for record in self.compliance_records
            if record.status == ComplianceStatus.COMPLIANT
        )
        non_compliant_layers = sum(
            1 for record in self.compliance_records
            if record.status == ComplianceStatus.NON_COMPLIANT
        )
        partially_compliant_layers = sum(
            1 for record in self.compliance_records
            if record.status == ComplianceStatus.PARTIALLY_COMPLIANT
        )
        
        # Complexity compliance
        complexity_compliant = sum(
            1 for record in self.compliance_records
            if record.complexity_bound and record.complexity_bound.status == ComplianceStatus.COMPLIANT
        )
        complexity_non_compliant = sum(
            1 for record in self.compliance_records
            if record.complexity_bound and record.complexity_bound.status == ComplianceStatus.NON_COMPLIANT
        )
        
        # Mathematical invariants
        total_invariants = sum(
            len(record.mathematical_invariants)
            for record in self.compliance_records
        )
        passed_invariants = sum(
            sum(1 for v in record.mathematical_invariants.values() if v)
            for record in self.compliance_records
        )
        
        report = {
            "summary": {
                "total_layers": total_layers,
                "compliant_layers": compliant_layers,
                "non_compliant_layers": non_compliant_layers,
                "partially_compliant_layers": partially_compliant_layers,
                "compliance_rate": compliant_layers / total_layers if total_layers > 0 else 0
            },
            "complexity_compliance": {
                "compliant": complexity_compliant,
                "non_compliant": complexity_non_compliant,
                "compliance_rate": complexity_compliant / (complexity_compliant + complexity_non_compliant)
                if (complexity_compliant + complexity_non_compliant) > 0 else 0
            },
            "mathematical_invariants": {
                "total": total_invariants,
                "passed": passed_invariants,
                "failed": total_invariants - passed_invariants,
                "pass_rate": passed_invariants / total_invariants if total_invariants > 0 else 0
            },
            "layer_details": [
                {
                    "layer": record.layer,
                    "theorem": record.theorem,
                    "status": record.status.value,
                    "conditions": f"{record.conditions_passed}/{record.conditions_checked}",
                    "complexity": record.complexity_bound.theoretical if record.complexity_bound else "N/A",
                    "complexity_status": record.complexity_bound.status.value if record.complexity_bound else "N/A",
                    "invariants": record.mathematical_invariants,
                    "notes": record.notes
                }
                for record in self.compliance_records
            ]
        }
        
        return report
    
    def get_non_compliant_theorems(self) -> List[TheoremCompliance]:
        """Get list of non-compliant theorems"""
        return [
            record for record in self.compliance_records
            if record.status == ComplianceStatus.NON_COMPLIANT
        ]
    
    def clear_records(self):
        """Clear all compliance records"""
        self.compliance_records.clear()
