"""
Feasibility Orchestrator - Core execution engine for seven-layer validation
Implements fail-fast termination and cross-layer metrics calculation
Enhanced with Stage 3 adapter, comprehensive logging, and error handling
"""

import time
import psutil
import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

from core.data_structures import (
    FeasibilityInput,
    FeasibilityOutput,
    LayerResult,
    ValidationStatus,
    FeasibilityConfig,
    CrossLayerMetrics,
    MathematicalProof
)

from core.stage3_adapter import Stage3Adapter, Stage3Data

from layers.layer_1_bcnf import BCNFValidator
from layers.layer_2_integrity import IntegrityValidator
from layers.layer_3_capacity import CapacityValidator
from layers.layer_4_temporal import TemporalValidator
from layers.layer_5_competency import CompetencyValidator
from layers.layer_6_conflict import ConflictValidator
from layers.layer_7_propagation import PropagationValidator

from utils.metrics_calculator import CrossLayerMetricsCalculator
from utils.report_generator import FeasibilityReportGenerator
from utils.logger import StructuredLogger
from utils.error_handler import ErrorHandler, ErrorSeverity

from validators.theorem_validator import TheoremComplianceChecker


class FeasibilityOrchestrator:
    """
    Orchestrates the seven-layer mathematical feasibility validation pipeline
    Based on Stage-4 FEASIBILITY CHECK theoretical foundations
    
    Enhanced features:
    - Stage 3 data adapter for proper input parsing
    - Comprehensive logging (JSON + console)
    - Error handling with detailed reports
    - Theorem compliance tracking
    - Performance monitoring (no artificial caps)
    """
    
    def __init__(
        self,
        config: FeasibilityConfig = None,
        structured_logger: Optional[StructuredLogger] = None,
        error_handler: Optional[ErrorHandler] = None
    ):
        self.config = config or FeasibilityConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize structured logger if provided
        self.structured_logger = structured_logger
        
        # Initialize error handler if provided
        self.error_handler = error_handler
        
        # Initialize Stage 3 adapter
        self.stage3_adapter: Optional[Stage3Adapter] = None
        self.stage3_data: Optional[Stage3Data] = None
        
        # Initialize validators
        self.validators = [
            BCNFValidator(self.config.layer_1_config),
            IntegrityValidator(self.config.layer_2_config),
            CapacityValidator(self.config.layer_3_config),
            TemporalValidator(self.config.layer_4_config),
            CompetencyValidator(self.config.layer_5_config),
            ConflictValidator(self.config.layer_6_config),
            PropagationValidator(self.config.layer_7_config)
        ]
        
        # Initialize utilities
        self.metrics_calculator = CrossLayerMetricsCalculator()
        self.report_generator = FeasibilityReportGenerator()
        
        # Initialize theorem compliance checker
        self.theorem_checker = TheoremComplianceChecker()
        
        self.logger.info("Feasibility Orchestrator initialized with 7 layers")
    
    def execute_feasibility_check(
        self,
        input_directory: Union[str, Path],
        output_directory: Union[str, Path]
    ) -> FeasibilityOutput:
        """
        Execute the complete seven-layer feasibility validation pipeline
        
        Args:
            input_directory: Directory containing Stage 3 compiled data
            output_directory: Directory to save feasibility results
            
        Returns:
            FeasibilityOutput: Complete feasibility analysis results
        """
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            # Log start
            if self.structured_logger:
                self.structured_logger.info(
                    "Starting Stage 4 feasibility check",
                    input_directory=str(input_directory),
                    output_directory=str(output_directory)
                )
            
            # Prepare input
            feasibility_input = FeasibilityInput(input_directory)
            
            # Ensure output directory exists
            output_path = Path(output_directory)
            output_path.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Starting feasibility check: {input_directory} -> {output_directory}")
            
            # Load Stage 3 data using adapter
            self.logger.info("Loading Stage 3 compiled data...")
            self.stage3_adapter = Stage3Adapter(Path(input_directory), self.error_handler)
            
            try:
                self.stage3_data = self.stage3_adapter.load_stage3_data()
                
                # Validate data completeness
                validation = self.stage3_adapter.validate_data_completeness(self.stage3_data)
                if not all(validation.values()):
                    missing = [k for k, v in validation.items() if not v]
                    error_msg = f"Stage 3 data incomplete: {missing}"
                    self.logger.error(error_msg)
                    if self.error_handler:
                        self.error_handler.handle_error(
                            Exception(error_msg),
                            layer="Stage3Adapter",
                            context={"validation": validation}
                        )
                    raise Exception(error_msg)
                
                # Get data summary
                summary = self.stage3_adapter.get_data_summary(self.stage3_data)
                self.logger.info(f"Loaded Stage 3 data: {summary}")
                
                if self.structured_logger:
                    self.structured_logger.info(
                        "Stage 3 data loaded successfully",
                        entities=len(self.stage3_data.l_raw),
                        graph_nodes=self.stage3_data.l_rel.number_of_nodes(),
                        graph_edges=self.stage3_data.l_rel.number_of_edges()
                    )
                
            except Exception as e:
                self.logger.error(f"Failed to load Stage 3 data: {str(e)}")
                if self.error_handler:
                    self.error_handler.handle_error(e, layer="Stage3Adapter")
                raise
            
            # Execute layers sequentially with fail-fast
            layer_results = []
            
            for i, validator in enumerate(self.validators, 1):
                layer_start = time.time()
                layer_memory_start = psutil.Process().memory_info().rss / 1024 / 1024
                
                # Log layer start
                if self.structured_logger:
                    self.structured_logger.log_layer_start(validator.layer_name, i)
                
                try:
                    self.logger.info(f"Executing Layer {i}: {validator.layer_name}")
                    
                    # Execute layer validation
                    result = validator.validate(feasibility_input)
                    layer_results.append(result)
                    
                    # Check for early termination
                    if not result.is_valid() and self.config.fail_fast:
                        self.logger.warning(f"Layer {i} failed: {result.message}")
                        if self.structured_logger:
                            self.structured_logger.warning(
                                f"Layer {i} failed - terminating pipeline",
                                layer=f"Layer{i}",
                                reason=result.message
                            )
                        break
                    
                    self.logger.info(f"Layer {i} completed: {result.status.value}")
                    
                except Exception as e:
                    self.logger.error(f"Layer {i} execution failed: {str(e)}")
                    
                    # Handle error
                    if self.error_handler:
                        self.error_handler.handle_error(
                            e,
                            layer=f"Layer{i}",
                            context={"validator": validator.layer_name}
                        )
                    
                    error_result = LayerResult(
                        layer_number=i,
                        layer_name=validator.layer_name,
                        status=ValidationStatus.ERROR,
                        message=f"Layer execution failed: {str(e)}",
                        details={"error": str(e), "exception_type": type(e).__name__}
                    )
                    layer_results.append(error_result)
                    
                    if self.config.fail_fast:
                        break
                
                finally:
                    layer_end = time.time()
                    layer_memory_end = psutil.Process().memory_info().rss / 1024 / 1024
                    
                    # Update timing and memory info
                    layer_results[-1].execution_time_ms = (layer_end - layer_start) * 1000
                    layer_results[-1].memory_used_mb = layer_memory_end - layer_memory_start
                    
                    # Log layer completion
                    if self.structured_logger:
                        self.structured_logger.log_layer_complete(
                            validator.layer_name,
                            i,
                            layer_results[-1].status.value,
                            layer_results[-1].execution_time_ms,
                            layer_results[-1].memory_used_mb
                        )
                    
                    # Check theorem compliance
                    if layer_results[-1].mathematical_proof:
                        # Extract conditions from mathematical proof
                        conditions_checked = len(layer_results[-1].mathematical_proof.conditions)
                        conditions_passed = conditions_checked if layer_results[-1].is_valid() else 0
                        
                        # Get mathematical invariants
                        invariants = self.theorem_checker.check_mathematical_invariants(
                            i,
                            layer_results[-1].details
                        )
                        
                        # Record compliance
                        self.theorem_checker.check_layer_compliance(
                            layer_number=i,
                            theorem=layer_results[-1].mathematical_proof.theorem,
                            execution_time_ms=layer_results[-1].execution_time_ms,
                            data_size=len(self.stage3_data.l_raw),
                            conditions_checked=conditions_checked,
                            conditions_passed=conditions_passed,
                            mathematical_invariants=invariants
                        )
            
            # Calculate cross-layer metrics if enabled
            cross_layer_metrics = None
            if self.config.enable_cross_layer_metrics:
                try:
                    cross_layer_metrics = self.metrics_calculator.calculate_metrics(
                        feasibility_input, layer_results
                    )
                except Exception as e:
                    self.logger.warning(f"Cross-layer metrics calculation failed: {str(e)}")
                    if self.error_handler:
                        self.error_handler.handle_error(e, layer="CrossLayerMetrics")
                    cross_layer_metrics = CrossLayerMetrics(
                        aggregate_load_ratio=0.0,
                        window_tightness_index=0.0,
                        conflict_density=0.0,
                        total_entities=0,
                        total_constraints=0
                    )
            
            # Determine overall feasibility
            is_feasible = all(result.is_valid() for result in layer_results)
            failure_reason = None
            mathematical_summary = None
            
            if not is_feasible:
                failed_layers = [r for r in layer_results if not r.is_valid()]
                failure_reason = f"Failed layers: {[r.layer_name for r in failed_layers]}"
                mathematical_summary = self._generate_mathematical_summary(failed_layers)
            
            # Calculate final metrics
            total_time = time.time() - start_time
            peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Create output
            output = FeasibilityOutput(
                is_feasible=is_feasible,
                layer_results=layer_results,
                cross_layer_metrics=cross_layer_metrics,
                total_execution_time_ms=total_time * 1000,
                peak_memory_mb=peak_memory,
                failure_reason=failure_reason,
                mathematical_summary=mathematical_summary
            )
            
            # Save results
            self._save_results(output, output_path)
            
            # Save error reports if any errors occurred
            if self.error_handler and self.error_handler.error_reports:
                self.error_handler.save_error_reports(output_path / "reports" / "error_reports")
            
            # Generate theorem compliance report
            if self.theorem_checker.compliance_records:
                compliance_report = self.theorem_checker.generate_compliance_report()
                import json
                with open(output_path / "theorem_compliance_report.json", 'w') as f:
                    json.dump(compliance_report, f, indent=2)
            
            self.logger.info(f"Feasibility check completed: {'FEASIBLE' if is_feasible else 'INFEASIBLE'}")
            
            if self.structured_logger:
                self.structured_logger.info(
                    "Feasibility check completed",
                    is_feasible=is_feasible,
                    total_time_ms=output.total_execution_time_ms,
                    peak_memory_mb=output.peak_memory_mb
                )
            
            return output
            
        except Exception as e:
            self.logger.error(f"Feasibility check failed: {str(e)}")
            if self.error_handler:
                self.error_handler.handle_error(e, severity=ErrorSeverity.CRITICAL)
            raise
    
    def _generate_mathematical_summary(self, failed_layers: List[LayerResult]) -> str:
        """Generate mathematical summary of feasibility violations"""
        summary_parts = []
        
        for layer in failed_layers:
            if layer.mathematical_proof:
                summary_parts.append(
                    f"Layer {layer.layer_number} ({layer.layer_name}): "
                    f"{layer.mathematical_proof.theorem} - {layer.mathematical_proof.conclusion}"
                )
            else:
                summary_parts.append(
                    f"Layer {layer.layer_number} ({layer.layer_name}): {layer.message}"
                )
        
        return " | ".join(summary_parts)
    
    def _save_results(self, output: FeasibilityOutput, output_path: Path) -> None:
        """Save feasibility results to output directory"""
        try:
            # Save main results
            output.save_to_json(output_path / "feasibility_results.json")
            
            # Generate and save report
            report = self.report_generator.generate_report(output)
            with open(output_path / "feasibility_report.html", 'w') as f:
                f.write(report)
            
            # Save metrics CSV
            if output.cross_layer_metrics:
                metrics_csv = self.metrics_calculator.export_metrics_csv(output.cross_layer_metrics)
                with open(output_path / "cross_layer_metrics.csv", 'w') as f:
                    f.write(metrics_csv)
            
            self.logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
            raise
