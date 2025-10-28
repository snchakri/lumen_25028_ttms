"""
Extended Schema Tables for Stage-2 Batching System
Implements invertibility, metrics, and foundation compliance tables

Per Solution to Stage-2.md and OR-Tools-CP-SAT-Stage2-Foundation.md
"""

from typing import Dict, List
import uuid


# OR-Tools Bridge Section 9: Batch Assignment Metrics
BATCH_ASSIGNMENT_METRICS_SCHEMA = {
    'assignment_id': 'UUID PRIMARY KEY',
    'tenant_id': 'UUID NOT NULL',
    'institution_id': 'UUID NOT NULL',
    'batch_id': 'UUID NOT NULL',
    'student_id': 'UUID NOT NULL',
    
    # Assignment Quality Metrics
    'individual_homogeneity_score': 'DECIMAL(8,4) NOT NULL',
    'course_overlap_percentage': 'DECIMAL(5,2) NOT NULL',
    'shift_preference_match': 'BOOLEAN NOT NULL',
    'language_compatibility_score': 'DECIMAL(5,2) NOT NULL',
    
    # Optimization Metadata
    'assignment_iteration': 'INTEGER NOT NULL',
    'local_optimality_score': 'DECIMAL(8,4) NOT NULL',
    'constraint_satisfaction_level': 'DECIMAL(5,2) NOT NULL',
    
    # Solver Metadata
    'solver_decision_confidence': 'DECIMAL(5,4) NOT NULL',
    'variable_branching_depth': 'INTEGER NOT NULL',
    'constraint_propagation_count': 'INTEGER NOT NULL',
    'solution_space_reduction_ratio': 'DECIMAL(8,6) NOT NULL',
    
    # Foundation Compliance Metrics
    'batch_size_optimality': 'DECIMAL(6,3) NOT NULL',
    'academic_homogeneity_contribution': 'DECIMAL(6,3) NOT NULL',
    'resource_utilization_impact': 'DECIMAL(6,3) NOT NULL',
    
    # Dynamic Parameter Tracking
    'parameter_configuration_hash': 'VARCHAR(64) NOT NULL',
    'threshold_adaptations': 'JSON',
    'weight_adjustments': 'JSON',
    
    # Audit and Performance
    'created_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
    'solver_runtime_ms': 'INTEGER NOT NULL',
    'memory_usage_kb': 'INTEGER NOT NULL'
}


# OR-Tools Bridge Section 9: Batch Solution Metadata
BATCH_SOLUTION_METADATA_SCHEMA = {
    'solution_id': 'UUID PRIMARY KEY',
    'tenant_id': 'UUID NOT NULL',
    'institution_id': 'UUID NOT NULL',
    
    # Execution Metadata
    'execution_timestamp': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
    'solver_version': 'VARCHAR(50) NOT NULL',
    'parameter_configuration': 'JSON NOT NULL',
    'input_data_hash': 'VARCHAR(64) NOT NULL',
    
    # Solution Quality
    'total_optimization_score': 'DECIMAL(12,6) NOT NULL',
    'objective_function_values': 'JSON NOT NULL',  # [f1, f2, f3]
    'pareto_optimality_rank': 'INTEGER',
    'solution_feasibility_level': 'DECIMAL(5,2) NOT NULL',
    
    # Performance Metrics
    'total_solver_runtime_ms': 'INTEGER NOT NULL',
    'peak_memory_usage_kb': 'INTEGER NOT NULL',
    'constraint_propagations': 'INTEGER NOT NULL',
    'search_tree_nodes': 'INTEGER NOT NULL',
    'restarts_count': 'INTEGER NOT NULL',
    
    # Problem Characteristics
    'student_count': 'INTEGER NOT NULL',
    'batch_count': 'INTEGER NOT NULL',
    'total_courses': 'INTEGER NOT NULL',
    'constraint_density': 'DECIMAL(8,6) NOT NULL',
    
    # Validation Results
    'hard_constraint_violations': 'INTEGER NOT NULL DEFAULT 0',
    'soft_constraint_penalty_score': 'DECIMAL(12,6) NOT NULL DEFAULT 0',
    'foundation_compliance_score': 'DECIMAL(5,2) NOT NULL'
}


# Solution Document: Ambiguity Resolution A1 - Batch Code Derivation
BATCH_CODE_DERIVATION_LOG_SCHEMA = {
    'log_id': 'UUID PRIMARY KEY',
    'batch_id': 'UUID NOT NULL',
    'input_hash': 'VARCHAR(64) NOT NULL',
    'generation_algorithm': 'VARCHAR(20) NOT NULL DEFAULT v2.0',
    'student_signature': 'TEXT NOT NULL',
    'parameter_signature': 'TEXT NOT NULL',
    'generated_code': 'VARCHAR(50) NOT NULL',
    'generation_timestamp': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
}


# Solution Document: Ambiguity Resolution A2 - Canonical Similarity Matrix
CANONICAL_SIMILARITY_MATRIX_SCHEMA = {
    'similarity_id': 'UUID PRIMARY KEY',
    'student_id_1': 'UUID NOT NULL',
    'student_id_2': 'UUID NOT NULL',
    'tenant_id': 'UUID NOT NULL',
    
    # Individual similarity components (deterministic)
    'course_jaccard_similarity': 'DECIMAL(8,6) NOT NULL',
    'credit_weight_similarity': 'DECIMAL(8,6) NOT NULL',
    'semester_alignment_similarity': 'DECIMAL(8,6) NOT NULL',
    'program_coherence_similarity': 'DECIMAL(8,6) NOT NULL',
    
    # Weighted composite similarity
    'composite_similarity': 'DECIMAL(8,6) NOT NULL',
    
    # Computational metadata for invertibility
    'computation_algorithm': 'VARCHAR(20) NOT NULL DEFAULT canonical_v2',
    'weight_vector': 'JSON NOT NULL',
    'computation_hash': 'VARCHAR(64) NOT NULL',
    'computed_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
}


# Solution Document: Complete Invertibility System - Transformation Audit
STAGE2_TRANSFORMATION_AUDIT_SCHEMA = {
    'audit_id': 'UUID PRIMARY KEY',
    'execution_id': 'UUID NOT NULL',
    'transformation_step': 'VARCHAR(50) NOT NULL',
    
    # Input state capture (for perfect reconstruction)
    'input_hash': 'VARCHAR(64) NOT NULL',
    'input_students': 'JSON NOT NULL',
    'input_courses': 'JSON NOT NULL',
    'input_parameters': 'JSON NOT NULL',
    
    # Transformation details
    'algorithm_version': 'VARCHAR(20) NOT NULL',
    'transformation_function': 'VARCHAR(100) NOT NULL',
    'canonical_ordering': 'JSON NOT NULL',
    
    # Output state capture
    'output_hash': 'VARCHAR(64) NOT NULL',
    'output_batches': 'JSON NOT NULL',
    'output_assignments': 'JSON NOT NULL',
    
    # Invertibility metadata
    'information_entropy_input': 'DECIMAL(12,6) NOT NULL',
    'information_entropy_output': 'DECIMAL(12,6) NOT NULL',
    'entropy_preservation_check': 'BOOLEAN',
    'transformation_timestamp': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
}


# Solution Document Section 3.2: Language Preferences
STUDENT_LANGUAGE_PREFERENCES_SCHEMA = {
    'preference_id': 'UUID PRIMARY KEY',
    'student_id': 'UUID NOT NULL',
    'tenant_id': 'UUID NOT NULL',
    
    'primary_instruction_language': 'CHAR(2) NOT NULL',
    'secondary_languages': 'CHAR(2)[] DEFAULT {}',
    'proficiency_levels': 'JSON NOT NULL',
    'preference_weight': 'DECIMAL(4,3) NOT NULL',
    'cultural_context': 'VARCHAR(50)',
    
    'validation_status': 'BOOLEAN DEFAULT FALSE',
    'validation_timestamp': 'TIMESTAMP',
    'effective_from': 'DATE DEFAULT CURRENT_DATE',
    'effective_to': 'DATE',
    'created_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
}


# Solution Document Section 3: Batch Optimization Metrics v2
BATCH_OPTIMIZATION_METRICS_V2_SCHEMA = {
    'metric_id': 'UUID PRIMARY KEY',
    'batch_id': 'UUID NOT NULL',
    'tenant_id': 'UUID NOT NULL',
    
    # Stage-2 Foundation multi-objective values
    'objective_f1_batch_size': 'DECIMAL(12,6) NOT NULL',
    'objective_f2_homogeneity': 'DECIMAL(12,6) NOT NULL',
    'objective_f3_resource_balance': 'DECIMAL(12,6) NOT NULL',
    
    # Foundation-required indices
    'homogeneity_index': 'DECIMAL(8,4) NOT NULL',
    'course_coherence_percentage': 'DECIMAL(5,2) NOT NULL',
    
    # Quality indicators
    'predicted_scheduling_complexity': 'DECIMAL(10,6)',
    'constraint_satisfaction_score': 'DECIMAL(8,4)',
    'optimization_iteration_count': 'INTEGER DEFAULT 0',
    
    # Foundation compliance validation
    'foundation_compliance_score': 'DECIMAL(5,2) NOT NULL',
    'coherence_threshold_satisfied': 'BOOLEAN',
    
    # Temporal tracking
    'optimization_timestamp': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
    'validation_timestamp': 'TIMESTAMP'
}


def get_all_extended_schemas() -> Dict[str, Dict]:
    """
    Get all extended schema definitions.
    
    Returns:
        Dictionary mapping table names to their schema definitions
    """
    return {
        'batch_assignment_metrics': BATCH_ASSIGNMENT_METRICS_SCHEMA,
        'batch_solution_metadata': BATCH_SOLUTION_METADATA_SCHEMA,
        'batch_code_derivation_log': BATCH_CODE_DERIVATION_LOG_SCHEMA,
        'canonical_similarity_matrix': CANONICAL_SIMILARITY_MATRIX_SCHEMA,
        'stage2_transformation_audit': STAGE2_TRANSFORMATION_AUDIT_SCHEMA,
        'student_language_preferences': STUDENT_LANGUAGE_PREFERENCES_SCHEMA,
        'batch_optimization_metrics_v2': BATCH_OPTIMIZATION_METRICS_V2_SCHEMA
    }

