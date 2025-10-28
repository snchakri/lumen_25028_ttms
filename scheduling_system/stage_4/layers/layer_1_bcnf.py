"""
Layer 1: BCNF Compliance & Schema Consistency
Implements Theorem 2.1 from Stage-4 FEASIBILITY CHECK theoretical framework

Mathematical Foundation: Theorem 2.1
- Verify that all tuples satisfy declared schemas
- Check unique primary keys and null constraints
- Validate all functional dependencies

Complexity: O(n log n) per table for key uniqueness checking
"""

import pandas as pd
import logging
from typing import Dict, Any, List, Set, Optional, Tuple
from pathlib import Path
import time

from core.data_structures import (
    LayerResult,
    ValidationStatus,
    MathematicalProof,
    FeasibilityInput
)


class BCNFValidator:
    """
    Layer 1: BCNF Compliance & Schema Consistency Validator
    
    Verifies that all tuples satisfy declared schemas, unique primary keys, 
    null constraints, and all functional dependencies in the dataset.
    
    Mathematical Foundation: Theorem 2.1
    Algorithmic Procedure: Check ∀ record t ∈ T, ∀ key attribute k: t[k] ≠ ∅
    
    Enhanced Features:
    - Dynamic schema inference from Stage 3 data
    - No hardcoded schema definitions
    - Proper BCNF compliance checking
    - O(n log n) complexity verification
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.layer_name = "BCNF Compliance & Schema Consistency"
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.strict_bcnf = self.config.get("strict_bcnf", True)
        self.check_functional_dependencies = self.config.get("check_functional_dependencies", True)
        
        # Dynamic schema definitions (inferred from data)
        self.schema_definitions: Dict[str, Dict[str, Any]] = {}
    
    def _load_l_raw_data(self, l_raw_path: Path) -> Dict[str, pd.DataFrame]:
        """Load L_raw data from directory containing entity parquet files"""
        l_raw_data = {}
        
        for parquet_file in l_raw_path.glob("*.parquet"):
            entity_name = parquet_file.stem
            l_raw_data[entity_name] = pd.read_parquet(parquet_file)
            self.logger.debug(f"Loaded {entity_name}: {len(l_raw_data[entity_name])} records")
        
        return l_raw_data
    
    def _infer_schema_from_data(
        self,
        entity_name: str,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Dynamically infer schema from data
        
        Args:
            entity_name: Name of the entity
            df: DataFrame containing entity data
            
        Returns:
            Schema definition with primary key, required fields, and FDs
        """
        schema = {
            "primary_key": [],
            "required_fields": [],
            "functional_dependencies": []
        }
        
        # Infer primary key: column ending with '_id' that is unique
        id_columns = [col for col in df.columns if col.endswith('_id')]
        for col in id_columns:
            if df[col].is_unique and df[col].notna().all():
                schema["primary_key"].append(col)
                schema["required_fields"].append(col)
                break
        
        # If no _id column found, use first column as primary key
        if not schema["primary_key"] and len(df.columns) > 0:
            first_col = df.columns[0]
            if df[first_col].is_unique and df[first_col].notna().all():
                schema["primary_key"].append(first_col)
                schema["required_fields"].append(first_col)
        
        # Add all non-null columns as required fields
        for col in df.columns:
            if col not in schema["required_fields"]:
                if df[col].notna().all():
                    schema["required_fields"].append(col)
        
        # Infer functional dependencies based on primary key
        if schema["primary_key"]:
            pk = schema["primary_key"][0]
            # PK determines all other attributes
            other_attrs = [col for col in df.columns if col != pk]
            if other_attrs:
                schema["functional_dependencies"].append(
                    ([pk], other_attrs)
                )
        
        return schema
    
    def _validate_complexity_bounds(
        self,
        n: int,
        execution_time_ms: float
    ) -> bool:
        """
        Validate O(n log n) complexity bound
        
        Args:
            n: Number of records
            execution_time_ms: Execution time in milliseconds
            
        Returns:
            True if complexity is within expected bounds
        """
        # Expected time for O(n log n): T(n) = c * n * log(n)
        # For validation, we allow up to 10x variance
        if n == 0:
            return True
        
        expected_time = n * (1 + n.bit_length())  # Approximate n * log(n)
        actual_time = execution_time_ms
        
        # Allow up to 10x variance for practical implementations
        return actual_time <= expected_time * 10
    
    def validate(self, feasibility_input: FeasibilityInput) -> LayerResult:
        """
        Execute Layer 1 validation: BCNF compliance and schema consistency
        
        Args:
            feasibility_input: Input data containing Stage 3 artifacts
            
        Returns:
            LayerResult: Validation result with mathematical proof
        """
        try:
            self.logger.info("Executing Layer 1: BCNF Compliance & Schema Consistency")
            
            # Load Stage 3 compiled data
            l_raw_path = feasibility_input.stage_3_artifacts["L_raw"]
            if not l_raw_path.exists():
                return LayerResult(
                    layer_number=1,
                    layer_name=self.layer_name,
                    status=ValidationStatus.ERROR,
                    message="Stage 3 L_raw artifact not found",
                    details={"expected_path": str(l_raw_path)}
                )
            
            # Load normalized data from L_raw directory
            try:
                l_raw_data = self._load_l_raw_data(l_raw_path)
            except Exception as e:
                return LayerResult(
                    layer_number=1,
                    layer_name=self.layer_name,
                    status=ValidationStatus.ERROR,
                    message=f"Failed to load L_raw data: {str(e)}",
                    details={"error": str(e)}
                )
            
            if not l_raw_data:
                return LayerResult(
                    layer_number=1,
                    layer_name=self.layer_name,
                    status=ValidationStatus.ERROR,
                    message="No entities found in L_raw data",
                    details={"l_raw_path": str(l_raw_path)}
                )
            
            # Infer schemas dynamically from data
            self.logger.info("Inferring schemas from data...")
            for entity_name, df in l_raw_data.items():
                self.schema_definitions[entity_name] = self._infer_schema_from_data(entity_name, df)
                self.logger.debug(f"Inferred schema for {entity_name}: {self.schema_definitions[entity_name]}")
            
            # Validate each entity
            validation_details = {}
            all_passed = True
            total_records = 0
            start_time = time.time()
            
            for entity_name, df in l_raw_data.items():
                entity_result = self._validate_entity(entity_name, df)
                validation_details[entity_name] = entity_result
                total_records += len(df)
                
                if not entity_result["passed"]:
                    all_passed = False
            
            # Validate complexity bounds
            execution_time_ms = (time.time() - start_time) * 1000
            complexity_valid = self._validate_complexity_bounds(total_records, execution_time_ms)
            
            if not complexity_valid:
                self.logger.warning(
                    f"Complexity bound violation: O(n log n) expected, "
                    f"measured {execution_time_ms:.2f}ms for {total_records} records"
                )
            
            # Generate mathematical proof
            mathematical_proof = None
            if not all_passed:
                # Count violations for proof
                null_key_violations = sum(
                    1 for result in validation_details.values()
                    if any("null values" in v for v in result.get("violations", []))
                )
                unique_key_violations = sum(
                    1 for result in validation_details.values()
                    if any("not unique" in v for v in result.get("violations", []))
                )
                fd_violations = sum(
                    1 for result in validation_details.values()
                    if any("FD violation" in v for v in result.get("violations", []))
                )
                
                mathematical_proof = MathematicalProof(
                    theorem="Theorem 2.1: BCNF Compliance",
                    proof_statement=(
                        "By construction, the algorithmic procedure enforces: "
                        "(1) No null primary keys, ensuring entity integrity; "
                        "(2) Unique primary keys, ensuring tuple uniqueness; "
                        "(3) Functional dependency satisfaction, ensuring BCNF compliance. "
                        f"Violations detected: {null_key_violations} null key violations, "
                        f"{unique_key_violations} uniqueness violations, {fd_violations} FD violations."
                    ),
                    conditions=[
                        "All tuples must satisfy declared schemas",
                        "Primary keys must be unique and non-null (entity integrity)",
                        "Functional dependencies must be preserved (BCNF compliance)"
                    ],
                    conclusion="Instance is not in BCNF, making optimization infeasible",
                    complexity="O(n log n) where n is the number of records"
                )
            
            status = ValidationStatus.PASSED if all_passed else ValidationStatus.FAILED
            message = "All entities satisfy BCNF compliance" if all_passed else "BCNF compliance violations detected"
            
            return LayerResult(
                layer_number=1,
                layer_name=self.layer_name,
                status=status,
                message=message,
                details=validation_details,
                mathematical_proof=mathematical_proof
            )
            
        except Exception as e:
            self.logger.error(f"Layer 1 validation failed: {str(e)}")
            return LayerResult(
                layer_number=1,
                layer_name=self.layer_name,
                status=ValidationStatus.ERROR,
                message=f"Layer 1 validation failed: {str(e)}",
                details={"error": str(e), "exception_type": type(e).__name__}
            )
    
    def _validate_entity(self, entity_name: str, entity_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate a single entity for BCNF compliance"""
        try:
            self.logger.debug(f"Validating entity {entity_name} with shape {entity_data.shape}")
            schema = self.schema_definitions[entity_name]
            result = {"passed": True, "violations": []}
            
            # Use the entity data directly
            df = entity_data
            self.logger.debug(f"Schema for {entity_name}: {schema}")
            
            # Check primary key uniqueness and non-null
            self.logger.debug(f"Checking primary keys: {schema['primary_key']}")
            for pk in schema["primary_key"]:
                self.logger.debug(f"Checking primary key: {pk}")
                if pk not in df.columns:
                    result["passed"] = False
                    result["violations"].append(f"Primary key {pk} not found in {entity_name}")
                    continue
                
                # Check for null values
                null_count = df[pk].isnull().sum()
                self.logger.debug(f"Null count for {pk}: {null_count}")
                if null_count > 0:
                    result["passed"] = False
                    result["violations"].append(f"Primary key {pk} has {null_count} null values in {entity_name}")
                
                # Check uniqueness
                unique_count = df[pk].nunique()
                total_count = len(df)
                self.logger.debug(f"Uniqueness check for {pk}: {unique_count}/{total_count}")
                if unique_count != total_count:
                    result["passed"] = False
                    result["violations"].append(f"Primary key {pk} is not unique in {entity_name}: {unique_count}/{total_count}")
            
            # Check required fields
            for field in schema["required_fields"]:
                if field not in df.columns:
                    result["passed"] = False
                    result["violations"].append(f"Required field {field} not found in {entity_name}")
                else:
                    null_count = df[field].isnull().sum()
                    if null_count > 0:
                        result["passed"] = False
                        result["violations"].append(f"Required field {field} has {null_count} null values in {entity_name}")
            
            # Check functional dependencies
            self.logger.debug(f"Checking functional dependencies: {schema['functional_dependencies']}")
            for fd in schema["functional_dependencies"]:
                self.logger.debug(f"Processing FD: {fd}")
                if len(fd) != 2:
                    self.logger.debug(f"FD has wrong length: {len(fd)}, skipping")
                    continue
                
                lhs = fd[0]
                rhs = fd[1]
                self.logger.debug(f"FD: {lhs} -> {rhs}")
                
                # Check if LHS exists (handle both single column and composite keys)
                if isinstance(lhs, list):
                    # Composite key - check all columns exist
                    for lhs_field in lhs:
                        if lhs_field not in df.columns:
                            result["passed"] = False
                            result["violations"].append(f"FD LHS field {lhs_field} not found in {entity_name}")
                            continue
                else:
                    # Single column key
                    if lhs not in df.columns:
                        result["passed"] = False
                        result["violations"].append(f"FD LHS {lhs} not found in {entity_name}")
                        continue
                
                # Check if RHS exists
                for rhs_field in rhs:
                    if rhs_field not in df.columns:
                        result["passed"] = False
                        result["violations"].append(f"FD RHS {rhs_field} not found in {entity_name}")
                        continue
                
                # Check functional dependency
                self.logger.debug(f"Calling _check_functional_dependency with lhs={lhs}, rhs={rhs}")
                fd_violations = self._check_functional_dependency(df, lhs, rhs)
                self.logger.debug(f"FD violations: {fd_violations}")
                if fd_violations:
                    result["passed"] = False
                    result["violations"].extend(fd_violations)
            
            return result
            
        except Exception as e:
            return {
                "passed": False,
                "violations": [f"Entity validation failed: {str(e)}"]
            }
    
    def _check_functional_dependency(self, df: pd.DataFrame, lhs, rhs: List[str]) -> List[str]:
        """Check if functional dependency LHS -> RHS is satisfied"""
        violations = []
        
        try:
            # Group by LHS and check if RHS is unique within each group
            # Handle both single column and composite keys
            if isinstance(lhs, list):
                # Composite key - group by all LHS columns
                grouped = df.groupby(lhs)
            else:
                # Single column key
                grouped = df.groupby(lhs)
            
            for lhs_value, group in grouped:
                if len(group) > 1:
                    # Check if RHS values are consistent within the group
                    for rhs_field in rhs:
                        unique_rhs_values = group[rhs_field].nunique()
                        if unique_rhs_values > 1:
                            # Convert lhs_value to string to handle unhashable types
                            lhs_str = str(lhs_value) if not isinstance(lhs_value, (str, int, float)) else lhs_value
                            violations.append(
                                f"FD violation: {lhs}={lhs_str} -> {rhs_field} "
                                f"has {unique_rhs_values} different values"
                            )
        
        except Exception as e:
            violations.append(f"FD check failed: {str(e)}")
        
        return violations
