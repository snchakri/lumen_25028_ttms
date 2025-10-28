"""
Mathematical Validation Framework

Provides rigorous mathematical validation including symbolic constraint
checking, statistical distribution tests, graph theory validation,
CSP solving, and formal verification.
"""

from .symbolic_math import (
    SymbolicMath,
    SymbolicConstraint,
    ConstraintType,
    LogicalOperator
)
from .constraint_validator import (
    ConstraintValidator,
    ValidationMode,
    ValidationResult,
    ValidationReport
)
from .proof_generator import (
    ProofGenerator,
    ProofCertificate,
    ProofStatus,
    ProofMethod,
    ProofStep
)
from .constraint_extractor import ConstraintExtractor, ConstraintDefinition
from .graph_validator import GraphValidator
from .statistical_validator import StatisticalValidator, DistributionType
from .csp_solver import CSPSolver, EnrollmentCSPConfig, RoomAssignmentCSPConfig
from .formal_verifier import FormalVerifier, TheoremLibrary, Theorem
from .validator_orchestrator import ValidatorOrchestrator, OrchestrationConfig, OrchestrationResult
from .adversarial_tester import AdversarialTester, AdversarialConfig

__all__ = [
    # Symbolic Math
    "SymbolicMath",
    "SymbolicConstraint",
    "ConstraintType",
    "LogicalOperator",
    # Validation
    "ConstraintValidator",
    "ValidationMode",
    "ValidationResult",
    "ValidationReport",
    # Proofs
    "ProofGenerator",
    "ProofCertificate",
    "ProofStatus",
    "ProofMethod",
    "ProofStep",
    # Extraction
    "ConstraintExtractor",
    "ConstraintDefinition",
    # Graphs & Statistics
    "GraphValidator",
    "StatisticalValidator",
    "DistributionType",
    # CSP
    "CSPSolver",
    "EnrollmentCSPConfig",
    "RoomAssignmentCSPConfig",
    # Formal verification
    "FormalVerifier",
    "TheoremLibrary",
    "Theorem",
    # Orchestrator
    "ValidatorOrchestrator",
    "OrchestrationConfig",
    "OrchestrationResult",
    # Adversarial testing
    "AdversarialTester",
    "AdversarialConfig",
]
