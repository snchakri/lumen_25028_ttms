"""
Formal Proof Generation

Generates formal proof certificates for mathematical theorems
and constraints, providing verifiable mathematical guarantees.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum
import json
import logging

from .symbolic_math import SymbolicConstraint

logger = logging.getLogger(__name__)


class ProofStatus(Enum):
    """Status of proof verification."""
    VERIFIED = "verified"
    FAILED = "failed"
    PARTIAL = "partial"
    PENDING = "pending"


class ProofMethod(Enum):
    """Methods used for proof generation."""
    DIRECT = "direct"
    CONTRADICTION = "contradiction"
    INDUCTION = "induction"
    EXHAUSTIVE = "exhaustive"
    SYMBOLIC = "symbolic"
    STATISTICAL = "statistical"
    CONSTRUCTION = "construction"


@dataclass
class ProofStep:
    """
    Single step in a formal proof.
    
    Attributes:
        step_number: Sequential step number
        statement: Mathematical statement for this step
        justification: Reason/rule used for this step
        references: Previous steps or axioms referenced
        result: Computed result if applicable
    """
    step_number: int
    statement: str
    justification: str
    references: List[str] = field(default_factory=list)
    result: Optional[Any] = None


@dataclass
class ProofCertificate:
    """
    Formal proof certificate for a theorem or constraint.
    
    Provides complete verifiable proof that can be audited
    and reproduced.
    
    Attributes:
        theorem_id: Unique theorem identifier
        theorem_statement: Mathematical statement being proved
        proof_method: Method used for proof
        proof_steps: Sequence of logical deductions
        conclusion: Final conclusion
        status: Verification status
        timestamp: When proof was generated
        configuration: Generation configuration used
        metadata: Additional proof metadata
    """
    theorem_id: str
    theorem_statement: str
    proof_method: ProofMethod
    proof_steps: List[ProofStep]
    conclusion: str
    status: ProofStatus
    timestamp: datetime
    configuration: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert proof certificate to dictionary."""
        return {
            "theorem_id": self.theorem_id,
            "theorem_statement": self.theorem_statement,
            "proof_method": self.proof_method.value,
            "proof_steps": [
                {
                    "step_number": step.step_number,
                    "statement": step.statement,
                    "justification": step.justification,
                    "references": step.references,
                    "result": step.result
                }
                for step in self.proof_steps
            ],
            "conclusion": self.conclusion,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "configuration": self.configuration,
            "metadata": self.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert proof certificate to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProofCertificate":
        """Create proof certificate from dictionary."""
        steps = [
            ProofStep(
                step_number=s["step_number"],
                statement=s["statement"],
                justification=s["justification"],
                references=s.get("references", []),
                result=s.get("result")
            )
            for s in data["proof_steps"]
        ]
        
        return cls(
            theorem_id=data["theorem_id"],
            theorem_statement=data["theorem_statement"],
            proof_method=ProofMethod(data["proof_method"]),
            proof_steps=steps,
            conclusion=data["conclusion"],
            status=ProofStatus(data["status"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            configuration=data["configuration"],
            metadata=data.get("metadata", {})
        )


class ProofGenerator:
    """
    Generates formal proof certificates for theorems and constraints.
    
    Provides mathematical verification and audit trails for
    generated test data.
    """
    
    def __init__(self):
        """Initialize proof generator."""
        self.certificates: List[ProofCertificate] = []
        logger.info("ProofGenerator initialized")
    
    def generate_constraint_proof(
        self,
        constraint: SymbolicConstraint,
        data_values: Dict[str, Any],
        config: Dict[str, Any]
    ) -> ProofCertificate:
        """
        Generate proof certificate for constraint satisfaction.
        
        Args:
            constraint: Symbolic constraint to prove
            data_values: Actual data values
            config: Generation configuration
            
        Returns:
            ProofCertificate
            
        Example:
            >>> pg = ProofGenerator()
            >>> certificate = pg.generate_constraint_proof(
            ...     constraint,
            ...     {'credit_hours': 3},
            ...     {'seed': 12345}
            ... )
        """
        steps = []
        step_num = 1
        
        # Step 1: State the constraint
        steps.append(ProofStep(
            step_number=step_num,
            statement=f"Constraint: {constraint.name}",
            justification="Foundation specification",
            references=[constraint.source],
            result=str(constraint.expression)
        ))
        step_num += 1
        
        # Step 2: State the data values
        steps.append(ProofStep(
            step_number=step_num,
            statement=f"Data values: {data_values}",
            justification="Generated data",
            references=[],
            result=data_values
        ))
        step_num += 1
        
        # Step 3: Substitute values
        try:
            from .symbolic_math import SymbolicMath
            sm = SymbolicMath()
            # Copy symbols from constraint
            for var in constraint.variables:
                if var not in sm.symbols:
                    sm.create_symbol(var)
            
            # Evaluate
            result = sm.evaluate_constraint(constraint.constraint_id, data_values)
            
            steps.append(ProofStep(
                step_number=step_num,
                statement=f"Substitute values into constraint expression",
                justification="Symbolic evaluation",
                references=[f"Step {step_num-1}"],
                result=result
            ))
            step_num += 1
            
            # Step 4: Conclusion
            if result:
                conclusion = f"Constraint '{constraint.name}' is SATISFIED"
                status = ProofStatus.VERIFIED
                steps.append(ProofStep(
                    step_number=step_num,
                    statement=conclusion,
                    justification="Evaluation result is True",
                    references=[f"Step {step_num-1}"],
                    result=True
                ))
            else:
                conclusion = f"Constraint '{constraint.name}' is VIOLATED"
                status = ProofStatus.FAILED
                steps.append(ProofStep(
                    step_number=step_num,
                    statement=conclusion,
                    justification="Evaluation result is False",
                    references=[f"Step {step_num-1}"],
                    result=False
                ))
        
        except Exception as e:
            logger.error(f"Error generating proof for {constraint.constraint_id}: {e}")
            conclusion = f"Proof generation failed: {str(e)}"
            status = ProofStatus.FAILED
            steps.append(ProofStep(
                step_number=step_num,
                statement=conclusion,
                justification="Error during evaluation",
                references=[],
                result=None
            ))
        
        # Create certificate
        certificate = ProofCertificate(
            theorem_id=constraint.constraint_id,
            theorem_statement=constraint.description,
            proof_method=ProofMethod.SYMBOLIC,
            proof_steps=steps,
            conclusion=conclusion,
            status=status,
            timestamp=datetime.now(),
            configuration=config,
            metadata={
                "constraint_type": constraint.constraint_type.value,
                "source": constraint.source,
                "variables": constraint.variables
            }
        )
        
        self.certificates.append(certificate)
        logger.info(f"Generated proof certificate for {constraint.constraint_id}: {status.value}")
        
        return certificate
    
    def generate_theorem_proof(
        self,
        theorem_id: str,
        theorem_statement: str,
        proof_method: ProofMethod,
        data: Dict[str, Any],
        config: Dict[str, Any]
    ) -> ProofCertificate:
        """
        Generate proof certificate for a general theorem.
        
        Args:
            theorem_id: Unique theorem identifier
            theorem_statement: Mathematical statement
            proof_method: Method to use for proof
            data: Data values for verification
            config: Generation configuration
            
        Returns:
            ProofCertificate
        """
        steps = []
        step_num = 1
        
        # Step 1: State the theorem
        steps.append(ProofStep(
            step_number=step_num,
            statement=f"Theorem: {theorem_statement}",
            justification="Theorem statement",
            references=[],
            result=None
        ))
        step_num += 1
        
        # Step 2: State the approach
        steps.append(ProofStep(
            step_number=step_num,
            statement=f"Proof by {proof_method.value}",
            justification="Proof strategy",
            references=[],
            result=None
        ))
        step_num += 1
        
        # Execute proof based on method
        proof_result = self._execute_theorem_proof(
            theorem_id, theorem_statement, proof_method, data, steps, step_num
        )
        
        certificate = ProofCertificate(
            theorem_id=theorem_id,
            theorem_statement=theorem_statement,
            proof_method=proof_method,
            proof_steps=proof_result["steps"],
            conclusion=proof_result["conclusion"],
            status=proof_result["status"],
            timestamp=datetime.now(),
            configuration=config,
            metadata=data
        )
        
        self.certificates.append(certificate)
        logger.info(f"Generated theorem proof certificate for {theorem_id}")
        
        return certificate
    
    def generate_batch_proof(
        self,
        constraints: List[SymbolicConstraint],
        data_values: List[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> List[ProofCertificate]:
        """
        Generate proof certificates for multiple constraints.
        
        Args:
            constraints: List of constraints
            data_values: List of data value dictionaries
            config: Generation configuration
            
        Returns:
            List of ProofCertificates
        """
        certificates = []
        
        for constraint in constraints:
            for data in data_values:
                # Check if data contains required variables
                if all(var in data for var in constraint.variables):
                    cert = self.generate_constraint_proof(constraint, data, config)
                    certificates.append(cert)
        
        logger.info(f"Generated {len(certificates)} proof certificates in batch")
        return certificates
    
    def get_certificate(self, theorem_id: str) -> Optional[ProofCertificate]:
        """
        Get proof certificate by theorem ID.
        
        Args:
            theorem_id: Theorem identifier
            
        Returns:
            ProofCertificate or None if not found
        """
        for cert in self.certificates:
            if cert.theorem_id == theorem_id:
                return cert
        return None
    
    def get_certificates_by_status(
        self,
        status: ProofStatus
    ) -> List[ProofCertificate]:
        """
        Get all certificates with specific status.
        
        Args:
            status: Proof status to filter by
            
        Returns:
            List of matching certificates
        """
        return [cert for cert in self.certificates if cert.status == status]
    
    def get_verification_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about generated proofs.
        
        Returns:
            Dictionary with statistics
        """
        total = len(self.certificates)
        verified = len(self.get_certificates_by_status(ProofStatus.VERIFIED))
        failed = len(self.get_certificates_by_status(ProofStatus.FAILED))
        partial = len(self.get_certificates_by_status(ProofStatus.PARTIAL))
        pending = len(self.get_certificates_by_status(ProofStatus.PENDING))
        
        return {
            "total_certificates": total,
            "verified": verified,
            "failed": failed,
            "partial": partial,
            "pending": pending,
            "verification_rate": (verified / total * 100.0) if total > 0 else 0.0
        }
    
    def export_certificates(self, file_path: str) -> None:
        """
        Export all certificates to JSON file.
        
        Args:
            file_path: Path to output file
        """
        data = {
            "generated_at": datetime.now().isoformat(),
            "total_certificates": len(self.certificates),
            "certificates": [cert.to_dict() for cert in self.certificates]
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported {len(self.certificates)} certificates to {file_path}")
    
    def clear(self) -> None:
        """Clear all generated certificates."""
        self.certificates.clear()
        logger.info("Cleared all proof certificates")
    
    def _execute_theorem_proof(
        self,
        theorem_id: str,
        theorem_statement: str,
        proof_method: ProofMethod,
        data: Dict[str, Any],
        initial_steps: List[ProofStep],
        step_num: int
    ) -> Dict[str, Any]:
        """
        Execute theorem-specific proof logic.
        
        Args:
            theorem_id: Theorem identifier
            theorem_statement: Theorem statement
            proof_method: Proof method to use
            data: Data for verification
            initial_steps: Proof steps so far
            step_num: Current step number
            
        Returns:
            Dictionary with steps, conclusion, and status
        """
        steps = list(initial_steps)
        
        # Dispatch to theorem-specific prover
        if theorem_id == "credit_limit_satisfaction":
            return self._prove_credit_limit(data, steps, step_num)
        elif theorem_id == "prerequisite_acyclicity":
            return self._prove_dag_property(data, steps, step_num)
        elif theorem_id == "room_capacity_adequacy":
            return self._prove_capacity_adequacy(data, steps, step_num)
        elif theorem_id == "faculty_coverage":
            return self._prove_faculty_coverage(data, steps, step_num)
        elif theorem_id == "schedule_feasibility":
            return self._prove_schedule_feasibility(data, steps, step_num)
        else:
            # Generic proof for unknown theorems
            steps.append(ProofStep(
                step_number=step_num,
                statement=f"Theorem {theorem_id} verification requires custom implementation",
                justification="Theorem-specific logic not yet implemented",
                references=[],
                result=None
            ))
            return {
                "steps": steps,
                "conclusion": f"Theorem {theorem_id} verification pending",
                "status": ProofStatus.PENDING
            }
    
    def _prove_credit_limit(
        self,
        data: Dict[str, Any],
        steps: List[ProofStep],
        step_num: int
    ) -> Dict[str, Any]:
        """Prove: For all students, total enrolled credits <= 27."""
        students = data.get("students", [])
        
        steps.append(ProofStep(
            step_number=step_num,
            statement=f"Examine all {len(students)} students",
            justification="Universal quantification over students",
            references=[],
            result={"student_count": len(students)}
        ))
        step_num += 1
        
        violations = []
        for student in students:
            total_credits = student.get("total_credits", 0)
            if total_credits > 27:
                violations.append({
                    "student_id": student.get("student_id"),
                    "credits": total_credits
                })
        
        if violations:
            steps.append(ProofStep(
                step_number=step_num,
                statement=f"Found {len(violations)} students exceeding credit limit",
                justification="Counter-examples found",
                references=[],
                result={"violations": violations}
            ))
            return {
                "steps": steps,
                "conclusion": f"Theorem FAILED: {len(violations)} violations found",
                "status": ProofStatus.FAILED
            }
        else:
            steps.append(ProofStep(
                step_number=step_num,
                statement="All students satisfy credit limit <= 27",
                justification="Exhaustive verification",
                references=[],
                result={"max_credits": max((s.get("total_credits", 0) for s in students), default=0)}
            ))
            return {
                "steps": steps,
                "conclusion": "Theorem VERIFIED: All students within credit limit",
                "status": ProofStatus.VERIFIED
            }
    
    def _prove_dag_property(
        self,
        data: Dict[str, Any],
        steps: List[ProofStep],
        step_num: int
    ) -> Dict[str, Any]:
        """Prove: Prerequisite relation forms a DAG."""
        try:
            import networkx as nx
            
            courses = data.get("courses", [])
            prerequisites = data.get("prerequisites", [])
            
            steps.append(ProofStep(
                step_number=step_num,
                statement=f"Construct prerequisite graph with {len(courses)} nodes and {len(prerequisites)} edges",
                justification="Graph construction from prerequisite relations",
                references=[],
                result={"nodes": len(courses), "edges": len(prerequisites)}
            ))
            step_num += 1
            
            # Build graph
            G = nx.DiGraph()
            G.add_nodes_from([c.get("course_id") for c in courses])
            G.add_edges_from([(p.get("prerequisite_id"), p.get("course_id")) for p in prerequisites])
            
            # Check DAG property
            is_dag = nx.is_directed_acyclic_graph(G)
            
            if not is_dag:
                cycles = list(nx.simple_cycles(G))
                steps.append(ProofStep(
                    step_number=step_num,
                    statement=f"Graph contains {len(cycles)} cycles",
                    justification="Cycle detection algorithm",
                    references=[],
                    result={"cycles": [list(c) for c in cycles[:5]]}  # First 5 cycles
                ))
                return {
                    "steps": steps,
                    "conclusion": f"Theorem FAILED: Graph is not a DAG ({len(cycles)} cycles found)",
                    "status": ProofStatus.FAILED
                }
            else:
                steps.append(ProofStep(
                    step_number=step_num,
                    statement="No cycles detected in prerequisite graph",
                    justification="NetworkX DAG verification algorithm",
                    references=[],
                    result={"is_dag": True}
                ))
                return {
                    "steps": steps,
                    "conclusion": "Theorem VERIFIED: Prerequisite graph is a DAG",
                    "status": ProofStatus.VERIFIED
                }
        except ImportError:
            steps.append(ProofStep(
                step_number=step_num,
                statement="NetworkX not available for graph verification",
                justification="Missing dependency",
                references=[],
                result=None
            ))
            return {
                "steps": steps,
                "conclusion": "Theorem verification PARTIAL: NetworkX required",
                "status": ProofStatus.PARTIAL
            }
    
    def _prove_capacity_adequacy(
        self,
        data: Dict[str, Any],
        steps: List[ProofStep],
        step_num: int
    ) -> Dict[str, Any]:
        """Prove: Total room capacity >= total student enrollment."""
        rooms = data.get("rooms", [])
        enrollments = data.get("enrollments", [])
        
        total_capacity = sum(r.get("capacity", 0) for r in rooms)
        total_enrollment = len(enrollments)
        
        steps.append(ProofStep(
            step_number=step_num,
            statement=f"Total room capacity: {total_capacity}, Total enrollments: {total_enrollment}",
            justification="Sum of room capacities and enrollment count",
            references=[],
            result={"total_capacity": total_capacity, "total_enrollment": total_enrollment}
        ))
        step_num += 1
        
        if total_capacity >= total_enrollment:
            steps.append(ProofStep(
                step_number=step_num,
                statement=f"Capacity {total_capacity} >= Enrollment {total_enrollment}",
                justification="Inequality verification",
                references=[],
                result={"surplus": total_capacity - total_enrollment}
            ))
            return {
                "steps": steps,
                "conclusion": f"Theorem VERIFIED: Adequate capacity (surplus: {total_capacity - total_enrollment})",
                "status": ProofStatus.VERIFIED
            }
        else:
            deficit = total_enrollment - total_capacity
            steps.append(ProofStep(
                step_number=step_num,
                statement=f"Capacity {total_capacity} < Enrollment {total_enrollment}",
                justification="Inequality violation",
                references=[],
                result={"deficit": deficit}
            ))
            return {
                "steps": steps,
                "conclusion": f"Theorem FAILED: Insufficient capacity (deficit: {deficit})",
                "status": ProofStatus.FAILED
            }
    
    def _prove_faculty_coverage(
        self,
        data: Dict[str, Any],
        steps: List[ProofStep],
        step_num: int
    ) -> Dict[str, Any]:
        """Prove: Every course has at least one qualified faculty."""
        courses = data.get("courses", [])
        faculty = data.get("faculty", [])
        competencies = data.get("competencies", [])
        
        steps.append(ProofStep(
            step_number=step_num,
            statement=f"Check faculty coverage for {len(courses)} courses",
            justification="Existential quantification over faculty for each course",
            references=[],
            result={"course_count": len(courses), "faculty_count": len(faculty)}
        ))
        step_num += 1
        
        # Build competency map: course_id -> [faculty with competency >= 4]
        comp_map: Dict[str, List[str]] = {}
        for comp in competencies:
            course_id = comp.get("course_id")
            faculty_id = comp.get("faculty_id")
            level = comp.get("competency_level", 0)
            if level >= 4:
                comp_map.setdefault(course_id, []).append(faculty_id)
        
        uncovered_courses = []
        for course in courses:
            course_id = course.get("course_id")
            if course_id not in comp_map or not comp_map[course_id]:
                uncovered_courses.append(course_id)
        
        if uncovered_courses:
            steps.append(ProofStep(
                step_number=step_num,
                statement=f"Found {len(uncovered_courses)} courses without qualified faculty",
                justification="Exhaustive check of competency mappings",
                references=[],
                result={"uncovered_courses": uncovered_courses[:10]}  # First 10
            ))
            return {
                "steps": steps,
                "conclusion": f"Theorem FAILED: {len(uncovered_courses)} courses lack qualified faculty",
                "status": ProofStatus.FAILED
            }
        else:
            steps.append(ProofStep(
                step_number=step_num,
                statement="All courses have at least one qualified faculty member",
                justification="Exhaustive verification of competency >= 4",
                references=[],
                result={"all_covered": True}
            ))
            return {
                "steps": steps,
                "conclusion": "Theorem VERIFIED: All courses have qualified faculty",
                "status": ProofStatus.VERIFIED
            }
    
    def _prove_schedule_feasibility(
        self,
        data: Dict[str, Any],
        steps: List[ProofStep],
        step_num: int
    ) -> Dict[str, Any]:
        """Prove: Generated schedule has no hard conflicts."""
        schedules = data.get("schedules", [])
        
        steps.append(ProofStep(
            step_number=step_num,
            statement=f"Check {len(schedules)} schedule entries for conflicts",
            justification="Pairwise constraint satisfaction check",
            references=[],
            result={"schedule_count": len(schedules)}
        ))
        step_num += 1
        
        conflicts = []
        
        # Check room conflicts: same room, overlapping time
        room_schedule: Dict[str, List[Dict[str, Any]]] = {}
        for sched in schedules:
            room_id = sched.get("room_id")
            room_schedule.setdefault(room_id, []).append(sched)
        
        for room_id, entries in room_schedule.items():
            for i in range(len(entries)):
                for j in range(i + 1, len(entries)):
                    # Simple overlap check (would need actual time comparison in real implementation)
                    conflicts.append({
                        "type": "room_conflict",
                        "room_id": room_id,
                        "entry1": entries[i].get("course_id"),
                        "entry2": entries[j].get("course_id")
                    })
        
        if conflicts:
            steps.append(ProofStep(
                step_number=step_num,
                statement=f"Found {len(conflicts)} scheduling conflicts",
                justification="Pairwise conflict detection",
                references=[],
                result={"conflicts": conflicts[:10]}  # First 10
            ))
            return {
                "steps": steps,
                "conclusion": f"Theorem FAILED: {len(conflicts)} schedule conflicts detected",
                "status": ProofStatus.FAILED
            }
        else:
            steps.append(ProofStep(
                step_number=step_num,
                statement="No scheduling conflicts detected",
                justification="Exhaustive pairwise verification",
                references=[],
                result={"conflict_free": True}
            ))
            return {
                "steps": steps,
                "conclusion": "Theorem VERIFIED: Schedule is conflict-free",
                "status": ProofStatus.VERIFIED
            }
    
    def __repr__(self) -> str:
        stats = self.get_verification_statistics()
        return (
            f"ProofGenerator("
            f"certificates={stats['total_certificates']}, "
            f"verified={stats['verified']}, "
            f"failed={stats['failed']})"
        )
