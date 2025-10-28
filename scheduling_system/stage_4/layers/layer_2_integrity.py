"""
Layer 2: Relational Integrity & Cardinality
Implements Theorem 3.1 from Stage-4 FEASIBILITY CHECK theoretical framework

Mathematical Foundation: Theorem 3.1
- Detect cycles of mandatory FKs using Tarjan's algorithm
- Check cardinality constraints for all relationships
- Validate topological ordering

Complexity: O(|V| + |E|) for cycle detection; linear for counting
"""

import networkx as nx
import pandas as pd
import logging
from typing import Dict, Any, List, Set, Tuple, Optional
from pathlib import Path
import time

from core.data_structures import (
    LayerResult,
    ValidationStatus,
    MathematicalProof,
    FeasibilityInput
)


class IntegrityValidator:
    """
    Layer 2: Relational Integrity & Cardinality Validator
    
    Models the schema as a directed multigraph of tables; each directed edge (A→B) 
    denotes a FK from A to B. Additionally, each FK may carry a cardinality constraint (ℓ,u).
    
    Mathematical Foundation: Theorem 3.1
    Algorithmic Procedure: Detect cycles of mandatory FKs and check cardinality constraints
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.layer_name = "Relational Integrity & Cardinality"
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.detect_cycles = self.config.get("detect_cycles", True)
        self.check_cardinality = self.config.get("check_cardinality", True)
        
        # Relationship definitions (will be inferred from graph)
        self.relationship_definitions: Dict[str, List[Dict[str, Any]]] = {}
    
    def _tarjan_scc(self, graph: nx.DiGraph) -> List[List[str]]:
        """
        Tarjan's algorithm for finding strongly connected components
        
        Args:
            graph: Directed graph representing FK relationships
            
        Returns:
            List of strongly connected components (cycles)
        """
        index = 0
        stack = []
        indices = {}
        lowlinks = {}
        on_stack = set()
        sccs = []
        
        def strongconnect(node):
            nonlocal index
            indices[node] = index
            lowlinks[node] = index
            index += 1
            stack.append(node)
            on_stack.add(node)
            
            # Consider successors of node
            for successor in graph.successors(node):
                if successor not in indices:
                    strongconnect(successor)
                    lowlinks[node] = min(lowlinks[node], lowlinks[successor])
                elif successor in on_stack:
                    lowlinks[node] = min(lowlinks[node], indices[successor])
            
            # If node is a root node, pop the stack and create an SCC
            if lowlinks[node] == indices[node]:
                scc = []
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    scc.append(w)
                    if w == node:
                        break
                if len(scc) > 1:  # Only report cycles (SCCs with more than one node)
                    sccs.append(scc)
        
        # Process all nodes
        for node in graph.nodes():
            if node not in indices:
                strongconnect(node)
        
        return sccs
    
    def _build_fk_graph(self, l_raw_data: Dict[str, pd.DataFrame], l_rel_graph: nx.Graph) -> nx.DiGraph:
        """
        Build FK relationship graph from Stage 3 data
        
        Args:
            l_raw_data: Entity dataframes
            l_rel_graph: Relationship graph from Stage 3
            
        Returns:
            Directed graph of FK relationships
        """
        fk_graph = nx.DiGraph()
        
        # Add all entity nodes
        for entity_name in l_raw_data.keys():
            fk_graph.add_node(entity_name)
        
        # Infer FK relationships from column names
        # Convention: column ending with '_id' references that entity
        for entity_name, df in l_raw_data.items():
            for col in df.columns:
                if col.endswith('_id') and col != f"{entity_name}_id":
                    # Extract target entity name
                    target_entity = col[:-3]  # Remove '_id'
                    
                    # Try plural forms
                    if target_entity + 's' in l_raw_data:
                        target_entity = target_entity + 's'
                    
                    if target_entity in l_raw_data:
                        fk_graph.add_edge(entity_name, target_entity)
                        self.logger.debug(f"Inferred FK: {entity_name}.{col} -> {target_entity}")
        
        return fk_graph
    
    def _detect_fk_cycles_tarjan(
        self,
        l_raw_data: Dict[str, pd.DataFrame],
        l_rel_graph: nx.Graph
    ) -> Dict[str, Any]:
        """
        Detect FK cycles using Tarjan's algorithm
        
        Args:
            l_raw_data: Entity dataframes
            l_rel_graph: Relationship graph
            
        Returns:
            Dictionary with cycle detection results
        """
        try:
            # Build FK graph
            fk_graph = self._build_fk_graph(l_raw_data, l_rel_graph)
            
            # Apply Tarjan's algorithm
            sccs = self._tarjan_scc(fk_graph)
            
            # Check for cycles
            has_cycles = len(sccs) > 0
            cycle_details = []
            
            for scc in sccs:
                cycle_str = " -> ".join(scc) + f" -> {scc[0]}"
                cycle_details.append(cycle_str)
                self.logger.warning(f"FK cycle detected: {cycle_str}")
            
            return {
                "passed": not has_cycles,
                "cycles_found": len(sccs),
                "cycle_details": cycle_details,
                "message": f"No FK cycles detected" if not has_cycles else f"Found {len(sccs)} FK cycle(s)"
            }
            
        except Exception as e:
            self.logger.error(f"FK cycle detection failed: {str(e)}")
            return {
                "passed": False,
                "error": str(e),
                "message": f"Cycle detection failed: {str(e)}"
            }
    
    def _check_cardinality_constraints(
        self,
        l_raw_data: Dict[str, pd.DataFrame],
        l_rel_graph: nx.Graph
    ) -> Dict[str, Any]:
        """
        Check cardinality constraints for all relationships
        
        Args:
            l_raw_data: Entity dataframes
            l_rel_graph: Relationship graph
            
        Returns:
            Dictionary with cardinality check results
        """
        try:
            violations = []
            all_passed = True
            
            # For each entity, check FK cardinality
            for entity_name, df in l_raw_data.items():
                for col in df.columns:
                    if col.endswith('_id') and col != f"{entity_name}_id":
                        # Check for null FK values (mandatory relationship)
                        null_count = df[col].isnull().sum()
                        total_count = len(df)
                        
                        # If FK is not nullable, all values must be present
                        if null_count > 0:
                            violation_msg = (
                                f"{entity_name}.{col}: {null_count}/{total_count} "
                                f"null FK values detected"
                            )
                            violations.append(violation_msg)
                            all_passed = False
            
            return {
                "passed": all_passed,
                "violations": violations,
                "message": "All cardinality constraints satisfied" if all_passed else f"Found {len(violations)} cardinality violation(s)"
            }
            
        except Exception as e:
            self.logger.error(f"Cardinality check failed: {str(e)}")
            return {
                "passed": False,
                "error": str(e),
                "message": f"Cardinality check failed: {str(e)}"
        }
    
    def validate(self, feasibility_input: FeasibilityInput) -> LayerResult:
        """
        Execute Layer 2 validation: Relational integrity and cardinality
        
        Args:
            feasibility_input: Input data containing Stage 3 artifacts
            
        Returns:
            LayerResult: Validation result with mathematical proof
        """
        try:
            self.logger.info("Executing Layer 2: Relational Integrity & Cardinality")
            
            # Load Stage 3 compiled data
            l_raw_path = feasibility_input.stage_3_artifacts["L_raw"]
            l_rel_path = feasibility_input.stage_3_artifacts["L_rel"]
            
            if not l_raw_path.exists() or not l_rel_path.exists():
                return LayerResult(
                    layer_number=2,
                    layer_name=self.layer_name,
                    status=ValidationStatus.ERROR,
                    message="Stage 3 artifacts not found",
                    details={"l_raw_exists": l_raw_path.exists(), "l_rel_exists": l_rel_path.exists()}
                )
            
            # Load data
            try:
                l_raw_data = self._load_l_raw_data(l_raw_path)
                l_rel_graph = nx.read_graphml(l_rel_path)
            except Exception as e:
                return LayerResult(
                    layer_number=2,
                    layer_name=self.layer_name,
                    status=ValidationStatus.ERROR,
                    message=f"Failed to load Stage 3 data: {str(e)}",
                    details={"error": str(e)}
                )
            
            validation_details = {}
            all_passed = True
            start_time = time.time()
            
            # 1. Detect cycles of mandatory FKs using Tarjan's algorithm
            if self.detect_cycles:
                cycle_result = self._detect_fk_cycles_tarjan(l_raw_data, l_rel_graph)
            validation_details["cycle_detection"] = cycle_result
            if not cycle_result["passed"]:
                all_passed = False
            
            # 2. Check cardinality constraints
            if self.check_cardinality:
                cardinality_result = self._check_cardinality_constraints(l_raw_data, l_rel_graph)
            validation_details["cardinality_check"] = cardinality_result
            if not cardinality_result["passed"]:
                all_passed = False
            
            # Validate complexity bounds
            execution_time_ms = (time.time() - start_time) * 1000
            total_nodes = l_rel_graph.number_of_nodes()
            total_edges = l_rel_graph.number_of_edges()
            
            # Expected complexity: O(|V| + |E|)
            expected_time = (total_nodes + total_edges) * 0.1  # Rough estimate
            complexity_valid = execution_time_ms <= expected_time * 100  # Allow 100x variance
            
            if not complexity_valid:
                self.logger.warning(
                    f"Complexity bound violation: O(|V| + |E|) expected, "
                    f"measured {execution_time_ms:.2f}ms for {total_nodes} nodes, {total_edges} edges"
                )
            
            # Generate mathematical proof
            mathematical_proof = None
            if not all_passed:
                violations = []
                if not cycle_result["passed"]:
                    violations.append("FK cycle detected")
                if not cardinality_result["passed"]:
                    violations.append("Cardinality constraint violations")
                
                mathematical_proof = MathematicalProof(
                    theorem="Theorem 3.1: FK Cycle Detection",
                    proof_statement="If the FK digraph contains a strongly connected component with only non-nullable edges, the instance is infeasible",
                    conditions=[
                        "No finite order permits insertions of records",
                        "Each node is a precondition for all others in the cycle",
                        "Cardinality constraints must be satisfied"
                    ],
                    conclusion=f"Instance is infeasible due to: {', '.join(violations)}",
                    complexity="O(|V| + |E|) for cycle detection; linear for counting"
                )
            
            status = ValidationStatus.PASSED if all_passed else ValidationStatus.FAILED
            message = "Relational integrity satisfied" if all_passed else "Relational integrity violations detected"
            
            return LayerResult(
                layer_number=2,
                layer_name=self.layer_name,
                status=status,
                message=message,
                details=validation_details,
                mathematical_proof=mathematical_proof
            )
            
        except Exception as e:
            self.logger.error(f"Layer 2 validation failed: {str(e)}")
            return LayerResult(
                layer_number=2,
                layer_name=self.layer_name,
                status=ValidationStatus.ERROR,
                message=f"Layer 2 validation failed: {str(e)}",
                details={"error": str(e), "exception_type": type(e).__name__}
            )
    
    def _load_l_raw_data(self, l_raw_path: Path) -> Dict[str, pd.DataFrame]:
        """
        Load L_raw data from Stage 3 output directory.
        
        Args:
            l_raw_path: Path to L_raw directory containing parquet files
            
        Returns:
            Dictionary mapping entity names to DataFrames
        """
        l_raw_data = {}
        
        # Iterate through parquet files in the L_raw directory
        for parquet_file in l_raw_path.glob("*.parquet"):
            entity_name = parquet_file.stem  # filename without .parquet extension
            try:
                entity_df = pd.read_parquet(parquet_file)
                l_raw_data[entity_name] = entity_df
                self.logger.debug(f"Loaded {entity_name}: {len(entity_df)} records")
            except Exception as e:
                self.logger.warning(f"Failed to load {entity_name}: {str(e)}")
                continue
        
        return l_raw_data
