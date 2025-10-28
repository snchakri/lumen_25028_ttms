"""
Formal Verifier and Theorem Library integration

Runs formal verification routines and compiles proof certificates for
core theorems defined in the design.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging

from .proof_generator import ProofGenerator, ProofCertificate, ProofMethod
from .symbolic_math import SymbolicMath, SymbolicConstraint

logger = logging.getLogger(__name__)


@dataclass
class Theorem:
    theorem_id: str
    statement: str
    metadata: Dict[str, Any]


class TheoremLibrary:
    """Registry of known theorems for verification."""

    def __init__(self):
        self.theorems: Dict[str, Theorem] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register all theorems from DESIGN_PART_3 foundation."""
        self.register(Theorem(
            theorem_id="credit_limit_satisfaction",
            statement="For all students, total enrolled credits <= 27",
            metadata={
                "type": "universal",
                "section": "5.2 Theorem Library",
                "priority": "critical"
            }
        ))
        self.register(Theorem(
            theorem_id="prerequisite_acyclicity",
            statement="Prerequisite relation forms a DAG",
            metadata={
                "type": "graph",
                "section": "5.2 Theorem Library",
                "priority": "critical"
            }
        ))
        self.register(Theorem(
            theorem_id="room_capacity_adequacy",
            statement="Total room capacity >= total student enrollment",
            metadata={
                "type": "inequality",
                "section": "5.2 Theorem Library",
                "priority": "high"
            }
        ))
        self.register(Theorem(
            theorem_id="faculty_coverage",
            statement="Every course has at least one qualified faculty",
            metadata={
                "type": "existential",
                "section": "5.2 Theorem Library",
                "priority": "critical"
            }
        ))
        self.register(Theorem(
            theorem_id="schedule_feasibility",
            statement="Generated schedule has no hard conflicts",
            metadata={
                "type": "pairwise",
                "section": "5.2 Theorem Library",
                "priority": "critical"
            }
        ))

    def register(self, theorem: Theorem) -> None:
        self.theorems[theorem.theorem_id] = theorem

    def get(self, theorem_id: str) -> Optional[Theorem]:
        return self.theorems.get(theorem_id)

    def list(self) -> List[str]:
        return list(self.theorems.keys())


class FormalVerifier:
    """Runs theorem verification and generates proof certificates."""

    def __init__(self, symbolic_math: Optional[SymbolicMath] = None):
        self.symbolic_math = symbolic_math or SymbolicMath()
        self.theorems = TheoremLibrary()
        self.proofs = ProofGenerator()

    def verify_constraint_sat(
        self,
        constraint: SymbolicConstraint,
        data_values: Dict[str, Any],
        config: Dict[str, Any]
    ) -> ProofCertificate:
        return self.proofs.generate_constraint_proof(constraint, data_values, config)

    def verify_theorem(
        self,
        theorem_id: str,
        data: Dict[str, Any],
        config: Dict[str, Any]
    ) -> ProofCertificate:
        """
        Verify a theorem against generated data.
        
        Args:
            theorem_id: Theorem identifier
            data: Generated data for verification
            config: Configuration used for generation
            
        Returns:
            ProofCertificate with verification result
            
        Raises:
            ValueError: If theorem ID is unknown
        """
        theorem = self.theorems.get(theorem_id)
        if not theorem:
            raise ValueError(f"Unknown theorem: {theorem_id}")
        
        # Select proof method based on theorem type
        proof_method = self._select_proof_method(theorem.metadata.get("type", "direct"))
        
        logger.info(f"Verifying theorem: {theorem_id} using {proof_method.value}")
        
        return self.proofs.generate_theorem_proof(
            theorem_id=theorem.theorem_id,
            theorem_statement=theorem.statement,
            proof_method=proof_method,
            data=data,
            config=config
        )
    
    def verify_all_theorems(
        self,
        data: Dict[str, Any],
        config: Dict[str, Any]
    ) -> List[ProofCertificate]:
        """
        Verify all registered theorems against generated data.
        
        Args:
            data: Generated data for verification
            config: Configuration used for generation
            
        Returns:
            List of proof certificates, one per theorem
        """
        certificates = []
        for theorem_id in self.theorems.list():
            try:
                cert = self.verify_theorem(theorem_id, data, config)
                certificates.append(cert)
            except Exception as e:
                logger.error(f"Failed to verify theorem {theorem_id}: {e}")
        
        logger.info(f"Verified {len(certificates)} theorems")
        return certificates
    
    def _select_proof_method(self, theorem_type: str) -> ProofMethod:
        """
        Select appropriate proof method based on theorem type.
        
        Args:
            theorem_type: Type of theorem
            
        Returns:
            Appropriate ProofMethod
        """
        method_map = {
            "universal": ProofMethod.DIRECT,
            "existential": ProofMethod.DIRECT,
            "graph": ProofMethod.CONSTRUCTION,
            "inequality": ProofMethod.DIRECT,
            "pairwise": ProofMethod.EXHAUSTIVE
        }
        return method_map.get(theorem_type, ProofMethod.DIRECT)

    def get_statistics(self) -> Dict[str, Any]:
        return self.proofs.get_verification_statistics()
