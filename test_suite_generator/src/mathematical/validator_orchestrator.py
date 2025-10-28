"""
Mathematical Validator Orchestrator

Coordinates execution of constraint extraction, symbolic validation,
statistical tests, graph validation, CSP solving, and formal verification.
"""

from dataclasses import dataclass
from typing import Any, Dict, List
import logging

from .constraint_extractor import ConstraintExtractor
from .constraint_validator import ConstraintValidator, ValidationMode
from .statistical_validator import StatisticalValidator
from .graph_validator import GraphValidator
from .formal_verifier import FormalVerifier
from .symbolic_math import SymbolicMath

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationConfig:
    mathematical_validation: bool = True
    adversarial_percentage: float = 0.0
    alpha: float = 0.05


@dataclass
class OrchestrationResult:
    constraints_checked: int
    constraint_pass_rate: float
    distributions_checked: int
    distributions_passed: int
    graphs_checked: int
    graphs_passed: int


class ValidatorOrchestrator:
    """
    Orchestrates all mathematical validation components.
    
    Coordinates:
    - Constraint extraction and symbolic validation
    - Statistical distribution testing
    - Graph theory validation
    - Formal theorem verification
    - CSP solving
    """
    
    def __init__(self):
        # Share one symbolic engine across components
        self.symbolic = SymbolicMath()
        self.validator = ConstraintValidator(self.symbolic)
        self.extractor = ConstraintExtractor(symbolic_math=self.symbolic)
        self.stats = StatisticalValidator()
        self.graphs = GraphValidator()
        self.verifier = FormalVerifier(self.symbolic)
        self._foundations_loaded = False
        logger.info("ValidatorOrchestrator initialized")

    def setup_foundations(self, foundations_dir: str) -> None:
        """
        Load foundation constraints from TOML files.
        
        Args:
            foundations_dir: Path to foundations directory
        """
        from pathlib import Path
        self.extractor.load_foundations(Path(foundations_dir))
        self.extractor.add_predefined_constraints()
        self.stats.register_predefined_distributions()
        self._foundations_loaded = True
        logger.info(f"Foundations loaded from {foundations_dir}")

    def validate_entities(
        self,
        entities: List[Dict[str, Any]],
        entity_type: str
    ) -> Dict[str, Any]:
        """
        Validate entities against symbolic constraints.
        
        Args:
            entities: List of entity dictionaries
            entity_type: Type of entities (e.g., 'students', 'courses')
            
        Returns:
            Validation report dictionary
        """
        if not self._foundations_loaded:
            logger.warning("Foundations not loaded, using only predefined constraints")
        
        report = self.validator.validate_batch(
            entities,
            entity_type,
            ValidationMode.POST_GENERATION
        )
        
        return {
            "total": report.total_constraints,
            "passed": report.passed,
            "failed": report.failed,
            "pass_rate": report.pass_rate,
            "violations": [
                {
                    "constraint_id": v.constraint_id,
                    "name": v.name,
                    "type": v.constraint_type.value
                }
                for v in report.violations
            ]
        }

    def validate_student_workload_distribution(
        self,
        credit_loads: List[float]
    ) -> Dict[str, Any]:
        """
        Validate student workload follows normal distribution.
        
        Args:
            credit_loads: List of student credit loads
            
        Returns:
            Distribution validation report
        """
        if not self._foundations_loaded:
            self.stats.register_predefined_distributions()
        
        report = self.stats.validate_continuous_distribution(
            credit_loads,
            "student_workload"
        )
        
        return {
            "distribution": report.distribution_name,
            "passed": report.passed,
            "tests": [r.message for r in report.test_results],
            "sample": report.sample_statistics,
        }

    def validate_prerequisite_graph(
        self,
        course_ids: List[str],
        edges: List[tuple[str, str]]
    ) -> Dict[str, Any]:
        """
        Validate prerequisite graph is a DAG with correct properties.
        
        Args:
            course_ids: List of course IDs
            edges: List of (prerequisite_id, course_id) tuples
            
        Returns:
            Graph validation report
        """
        gname = "prerequisites"
        self.graphs.create_graph(gname)  # type: ignore[call-arg]
        self.graphs.add_nodes(gname, course_ids)  # type: ignore[call-arg]
        self.graphs.add_edges(gname, edges)  # type: ignore[call-arg]
        result = self.graphs.validate_prerequisite_graph(
            gname,
            max_depth=4,
            max_prerequisites=3
        )
        return {
            "passed": result.passed,
            "violations": result.violations,
            "metrics": result.metrics.__dict__,
        }
    
    def verify_theorems(
        self,
        data: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Verify all registered theorems against generated data.
        
        Args:
            data: Generated data (students, courses, etc.)
            config: Generation configuration
            
        Returns:
            Theorem verification results
        """
        certificates = self.verifier.verify_all_theorems(data, config)
        
        return {
            "total_theorems": len(certificates),
            "verified": sum(1 for c in certificates if c.status.value == "verified"),
            "failed": sum(1 for c in certificates if c.status.value == "failed"),
            "partial": sum(1 for c in certificates if c.status.value == "partial"),
            "certificates": [
                {
                    "theorem_id": c.theorem_id,
                    "status": c.status.value,
                    "conclusion": c.conclusion,
                    "steps": len(c.proof_steps)
                }
                for c in certificates
            ]
        }
    
    def run_full_validation(
        self,
        data: Dict[str, Any],
        config: Dict[str, Any]
    ) -> OrchestrationResult:
        """
        Run complete validation suite on generated data.
        
        Args:
            data: Generated data with all entity types
            config: Generation configuration
            
        Returns:
            Comprehensive validation results
        """
        logger.info("Starting full validation suite")
        
        # Constraint validation
        constraints_checked = 0
        constraints_passed = 0
        for entity_type, entities in data.items():
            if isinstance(entities, list) and entities:
                result = self.validate_entities(entities, entity_type)
                constraints_checked += result["total"]
                constraints_passed += result["passed"]
        
        # Distribution validation
        distributions_checked = 0
        distributions_passed = 0
        if "students" in data and data["students"]:
            credit_loads = [s.get("total_credits", 0) for s in data["students"]]
            dist_result = self.validate_student_workload_distribution(credit_loads)
            distributions_checked += 1
            distributions_passed += 1 if dist_result["passed"] else 0
        
        # Graph validation
        graphs_checked = 0
        graphs_passed = 0
        if "prerequisites" in data and data["prerequisites"]:
            course_ids = list(set(
                [p.get("course_id") for p in data["prerequisites"]] +
                [p.get("prerequisite_id") for p in data["prerequisites"]]
            ))
            edges = [
                (p.get("prerequisite_id"), p.get("course_id"))
                for p in data["prerequisites"]
            ]
            graph_result = self.validate_prerequisite_graph(course_ids, edges)
            graphs_checked += 1
            graphs_passed += 1 if graph_result["passed"] else 0
        
        # Theorem verification
        theorem_results = self.verify_theorems(data, config)
        
        result = OrchestrationResult(
            constraints_checked=constraints_checked,
            constraint_pass_rate=(constraints_passed / constraints_checked * 100.0) if constraints_checked > 0 else 0.0,
            distributions_checked=distributions_checked,
            distributions_passed=distributions_passed,
            graphs_checked=graphs_checked,
            graphs_passed=graphs_passed
        )
        
        logger.info(
            f"Full validation complete: {constraints_passed}/{constraints_checked} constraints passed, "
            f"{distributions_passed}/{distributions_checked} distributions passed, "
            f"{graphs_passed}/{graphs_checked} graphs passed"
        )
        
        return result
