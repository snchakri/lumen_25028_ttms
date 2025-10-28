"""
Custom Exception Classes for DEAP Solver Family

Defines structured exception hierarchy for different error types
with foundation-compliant error handling.

Author: LUMEN Team [TEAM-ID: 93912]
"""

from typing import Dict, Any, Optional, List


class ValidationError(Exception):
    """Base validation error class."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "VALIDATION_ERROR",
        context: Optional[Dict[str, Any]] = None,
        foundation_section: Optional[str] = None
    ):
        """
        Initialize validation error.
        
        Args:
            message: Human-readable error message
            error_code: Structured error code
            context: Additional error context
            foundation_section: Related foundation section
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.foundation_section = foundation_section
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "foundation_section": self.foundation_section
        }


class InputValidationError(ValidationError):
    """Input data validation error."""
    
    def __init__(
        self,
        message: str,
        invalid_fields: Optional[List[str]] = None,
        missing_files: Optional[List[str]] = None,
        schema_violations: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize input validation error.
        
        Args:
            message: Error message
            invalid_fields: List of invalid field names
            missing_files: List of missing file paths
            schema_violations: List of schema violation descriptions
            **kwargs: Additional arguments for base class
        """
        context = kwargs.get('context', {})
        context.update({
            "invalid_fields": invalid_fields or [],
            "missing_files": missing_files or [],
            "schema_violations": schema_violations or []
        })
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'INPUT_VALIDATION_ERROR')
        
        super().__init__(message, **kwargs)
        
        self.invalid_fields = invalid_fields or []
        self.missing_files = missing_files or []
        self.schema_violations = schema_violations or []


class SolverError(ValidationError):
    """Solver execution error."""
    
    def __init__(
        self,
        message: str,
        solver_type: Optional[str] = None,
        solver_parameters: Optional[Dict[str, Any]] = None,
        convergence_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize solver error.
        
        Args:
            message: Error message
            solver_type: Type of solver that failed
            solver_parameters: Solver parameters at time of failure
            convergence_info: Convergence information if available
            **kwargs: Additional arguments for base class
        """
        context = kwargs.get('context', {})
        context.update({
            "solver_type": solver_type,
            "solver_parameters": solver_parameters or {},
            "convergence_info": convergence_info or {}
        })
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'SOLVER_ERROR')
        
        super().__init__(message, **kwargs)
        
        self.solver_type = solver_type
        self.solver_parameters = solver_parameters or {}
        self.convergence_info = convergence_info or {}


class EncodingError(ValidationError):
    """Genotype/Phenotype encoding error."""
    
    def __init__(
        self,
        message: str,
        encoding_type: Optional[str] = None,
        genotype_data: Optional[Any] = None,
        phenotype_data: Optional[Any] = None,
        bijection_test_failed: bool = False,
        **kwargs
    ):
        """
        Initialize encoding error.
        
        Args:
            message: Error message
            encoding_type: Type of encoding (direct, permutation, integer)
            genotype_data: Genotype data that caused error
            phenotype_data: Phenotype data that caused error
            bijection_test_failed: Whether bijection test failed
            **kwargs: Additional arguments for base class
        """
        context = kwargs.get('context', {})
        context.update({
            "encoding_type": encoding_type,
            "genotype_data": str(genotype_data) if genotype_data is not None else None,
            "phenotype_data": str(phenotype_data) if phenotype_data is not None else None,
            "bijection_test_failed": bijection_test_failed
        })
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'ENCODING_ERROR')
        kwargs['foundation_section'] = kwargs.get('foundation_section', 'Definition_2.2_Genotype_Encoding')
        
        super().__init__(message, **kwargs)
        
        self.encoding_type = encoding_type
        self.genotype_data = genotype_data
        self.phenotype_data = phenotype_data
        self.bijection_test_failed = bijection_test_failed


class FitnessError(ValidationError):
    """Fitness evaluation error."""
    
    def __init__(
        self,
        message: str,
        fitness_components: Optional[List[str]] = None,
        invalid_values: Optional[Dict[str, Any]] = None,
        constraint_violations: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize fitness error.
        
        Args:
            message: Error message
            fitness_components: List of fitness component names
            invalid_values: Dictionary of invalid fitness values
            constraint_violations: List of constraint violations
            **kwargs: Additional arguments for base class
        """
        context = kwargs.get('context', {})
        context.update({
            "fitness_components": fitness_components or [],
            "invalid_values": invalid_values or {},
            "constraint_violations": constraint_violations or []
        })
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'FITNESS_ERROR')
        kwargs['foundation_section'] = kwargs.get('foundation_section', 'Definition_2.4_Fitness_Function')
        
        super().__init__(message, **kwargs)
        
        self.fitness_components = fitness_components or []
        self.invalid_values = invalid_values or {}
        self.constraint_violations = constraint_violations or []


class ConstraintError(ValidationError):
    """Constraint handling error."""
    
    def __init__(
        self,
        message: str,
        constraint_type: Optional[str] = None,
        violated_constraints: Optional[List[str]] = None,
        constraint_density: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize constraint error.
        
        Args:
            message: Error message
            constraint_type: Type of constraint (hard, soft)
            violated_constraints: List of violated constraint names
            constraint_density: Constraint density measure
            **kwargs: Additional arguments for base class
        """
        context = kwargs.get('context', {})
        context.update({
            "constraint_type": constraint_type,
            "violated_constraints": violated_constraints or [],
            "constraint_density": constraint_density
        })
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'CONSTRAINT_ERROR')
        kwargs['foundation_section'] = kwargs.get('foundation_section', 'Section_9_Constraint_Handling')
        
        super().__init__(message, **kwargs)
        
        self.constraint_type = constraint_type
        self.violated_constraints = violated_constraints or []
        self.constraint_density = constraint_density


class ConfigurationError(ValidationError):
    """Configuration error."""
    
    def __init__(
        self,
        message: str,
        config_section: Optional[str] = None,
        invalid_parameters: Optional[List[str]] = None,
        parameter_bounds: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize configuration error.
        
        Args:
            message: Error message
            config_section: Configuration section with error
            invalid_parameters: List of invalid parameter names
            parameter_bounds: Dictionary of parameter bounds
            **kwargs: Additional arguments for base class
        """
        context = kwargs.get('context', {})
        context.update({
            "config_section": config_section,
            "invalid_parameters": invalid_parameters or [],
            "parameter_bounds": parameter_bounds or {}
        })
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'CONFIGURATION_ERROR')
        
        super().__init__(message, **kwargs)
        
        self.config_section = config_section
        self.invalid_parameters = invalid_parameters or []
        self.parameter_bounds = parameter_bounds or {}


class ConvergenceError(ValidationError):
    """Convergence failure error."""
    
    def __init__(
        self,
        message: str,
        generations_completed: Optional[int] = None,
        best_fitness: Optional[float] = None,
        convergence_criteria: Optional[Dict[str, Any]] = None,
        diversity_metrics: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """
        Initialize convergence error.
        
        Args:
            message: Error message
            generations_completed: Number of generations completed
            best_fitness: Best fitness achieved
            convergence_criteria: Convergence criteria used
            diversity_metrics: Population diversity metrics
            **kwargs: Additional arguments for base class
        """
        context = kwargs.get('context', {})
        context.update({
            "generations_completed": generations_completed,
            "best_fitness": best_fitness,
            "convergence_criteria": convergence_criteria or {},
            "diversity_metrics": diversity_metrics or {}
        })
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'CONVERGENCE_ERROR')
        kwargs['foundation_section'] = kwargs.get('foundation_section', 'Section_13_Performance_Analysis')
        
        super().__init__(message, **kwargs)
        
        self.generations_completed = generations_completed
        self.best_fitness = best_fitness
        self.convergence_criteria = convergence_criteria or {}
        self.diversity_metrics = diversity_metrics or {}


class OutputError(ValidationError):
    """Output generation error."""
    
    def __init__(
        self,
        message: str,
        output_type: Optional[str] = None,
        file_path: Optional[str] = None,
        validation_failures: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize output error.
        
        Args:
            message: Error message
            output_type: Type of output (CSV, JSON, Parquet)
            file_path: File path that failed
            validation_failures: List of validation failures
            **kwargs: Additional arguments for base class
        """
        context = kwargs.get('context', {})
        context.update({
            "output_type": output_type,
            "file_path": file_path,
            "validation_failures": validation_failures or []
        })
        kwargs['context'] = context
        kwargs['error_code'] = kwargs.get('error_code', 'OUTPUT_ERROR')
        
        super().__init__(message, **kwargs)
        
        self.output_type = output_type
        self.file_path = file_path
        self.validation_failures = validation_failures or []

