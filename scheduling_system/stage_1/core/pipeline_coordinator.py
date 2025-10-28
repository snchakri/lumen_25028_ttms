"""
Pipeline coordinator for orchestrating all 7 validation stages.

Implements complete validation pipeline with sequential execution,
error collection, metrics computation, and comprehensive reporting.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
from ..models.validation_types import ValidationResult, ValidationStatus, ErrorSeverity
from ..validators.stage1_syntactic import Stage1SyntacticValidator
from ..validators.stage2_structural import Stage2StructuralValidator
from ..validators.stage3_referential import Stage3ReferentialValidator
from ..validators.stage4_semantic import Stage4SemanticValidator
from ..validators.stage5_temporal import Stage5TemporalValidator
from ..validators.stage6_cross_table import Stage6CrossTableValidator
from ..validators.stage7_domain import Stage7DomainValidator
from ..processors.file_processor import FileProcessor
from ..parsers.csv_parser import CSVParser
from ..errors.error_collector import ErrorCollector
from ..logging_system.log_coordinator import LogCoordinator
from ..metrics.quality_metrics import QualityMetricsComputer
from ..metrics.complexity_analyzer import ComplexityAnalyzer
from ..metrics.statistical_summary import StatisticalSummary
from ..output.status_reporter import StatusReporter
from ..output.metrics_writer import MetricsWriter
from ..output.report_generator import ReportGenerator


class PipelineCoordinator:
    """
    Orchestrate all 7 validation stages sequentially.
    
    Implements the complete validation pipeline per theoretical foundations:
    - Sequential execution (Stages 1→7)
    - Error collection and aggregation
    - Metrics computation
    - Comprehensive reporting
    """
    
    def __init__(self, input_dir: Path, output_dir: Path, log_dir: Path):
        """
        Initialize pipeline coordinator.
        
        Args:
            input_dir: Directory containing input CSV files
            output_dir: Directory for output reports and metrics
            log_dir: Directory for log files
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.file_processor = FileProcessor(self.input_dir)
        self.parser = CSVParser()
        self.error_collector = ErrorCollector()
        self.logger = None
        
        # Initialize metrics and output systems
        self.quality_computer = QualityMetricsComputer()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.statistical_summary = StatisticalSummary()
        self.status_reporter = StatusReporter()
        self.metrics_writer = MetricsWriter(self.output_dir)
        self.report_generator = ReportGenerator(self.output_dir)
        
        # Initialize validators
        self.stage1_validator = Stage1SyntacticValidator()
        self.stage2_validator = Stage2StructuralValidator()
        self.stage3_validator = Stage3ReferentialValidator()
        self.stage4_validator = Stage4SemanticValidator()
        self.stage5_validator = Stage5TemporalValidator()
        self.stage6_validator = Stage6CrossTableValidator()
        self.stage7_validator = Stage7DomainValidator()
        
        # Pipeline state
        self.parsed_data = {}  # {filename: list of row dicts}
        self.stage_results = []
    
    def execute(self) -> ValidationResult:
        """
        Execute complete validation pipeline.
        
        Returns:
            ValidationResult with overall status and detailed reports
        """
        # Initialize logging
        self.logger = LogCoordinator(self.log_dir, console_verbose=True)
        self.logger.start()
        self.logger.info("Starting Stage-1 Input Validation Pipeline")
        
        try:
            # Step 1: Scan and validate file presence
            self.logger.info("Step 1: Scanning input directory for CSV files")
            success, errors, warnings = self.file_processor.scan_files()
            
            if not success:
                self.logger.critical(f"File scan failed: {', '.join(errors)}")
                return self._create_failure_result(', '.join(errors))
            
            self.logger.success(f"Found {len(self.file_processor.present_files)} CSV files")
            
            # Step 2: Parse all CSV files
            self.logger.info("Step 2: Parsing CSV files")
            file_paths = [self.file_processor.input_dir / filename for filename in self.file_processor.present_files]
            parse_errors = self._parse_all_files(file_paths)
            
            if parse_errors:
                self.logger.error(f"Parsing failed for {len(parse_errors)} files")
                return self._create_failure_result("CSV parsing failed", parse_errors)
            
            self.logger.success(f"Successfully parsed {len(self.parsed_data)} files")
            
            # Step 3: Execute validation stages sequentially
            self.logger.start_progress(total_stages=7)
            
            # Stage 1: Syntactic Validation (per-file)
            self.logger.start_stage(1, "Syntactic Validation")
            stage1_results = self._execute_stage1(file_paths)
            self.stage_results.extend(stage1_results)
            self._collect_stage_errors(stage1_results)
            stage1_status = stage1_results[0].status if stage1_results else ValidationStatus.FAIL
            stage1_exec_time = stage1_results[0].execution_time_seconds if stage1_results else 0.0
            self.logger.complete_stage(1, "Syntactic Validation", stage1_status.value, stage1_exec_time)
            
            # Stage 2: Structural Validation (per-file)
            self.logger.start_stage(2, "Structural Validation")
            stage2_results = self._execute_stage2(file_paths)
            self.stage_results.extend(stage2_results)
            self._collect_stage_errors(stage2_results)
            stage2_status = stage2_results[0].status if stage2_results else ValidationStatus.FAIL
            stage2_exec_time = stage2_results[0].execution_time_seconds if stage2_results else 0.0
            self.logger.complete_stage(2, "Structural Validation", stage2_status.value, stage2_exec_time)
            
            # Stages 3-7: Cross-file validation (use all parsed data)
            self.logger.start_stage(3, "Referential Integrity")
            stage3_result = self.stage3_validator.validate(self.parsed_data)
            self.stage_results.append(stage3_result)
            self._collect_stage_errors([stage3_result])
            self.logger.complete_stage(3, "Referential Integrity", stage3_result.status.value, stage3_result.execution_time_seconds)
            
            self.logger.start_stage(4, "Semantic Validation")
            stage4_result = self.stage4_validator.validate(self.parsed_data)
            self.stage_results.append(stage4_result)
            self._collect_stage_errors([stage4_result])
            self.logger.complete_stage(4, "Semantic Validation", stage4_result.status.value, stage4_result.execution_time_seconds)
            
            self.logger.start_stage(5, "Temporal Consistency")
            stage5_result = self.stage5_validator.validate(self.parsed_data)
            self.stage_results.append(stage5_result)
            self._collect_stage_errors([stage5_result])
            self.logger.complete_stage(5, "Temporal Consistency", stage5_result.status.value, stage5_result.execution_time_seconds)
            
            self.logger.start_stage(6, "Cross-Table Consistency")
            stage6_result = self.stage6_validator.validate(self.parsed_data)
            self.stage_results.append(stage6_result)
            self._collect_stage_errors([stage6_result])
            self.logger.complete_stage(6, "Cross-Table Consistency", stage6_result.status.value, stage6_result.execution_time_seconds)
            
            self.logger.start_stage(7, "Domain Compliance")
            stage7_result = self.stage7_validator.validate(self.parsed_data)
            self.stage_results.append(stage7_result)
            self._collect_stage_errors([stage7_result])
            self.logger.complete_stage(7, "Domain Compliance", stage7_result.status.value, stage7_result.execution_time_seconds)
            
            self.logger.stop_progress()
            
            # Step 4: Compute overall status
            overall_status = self._compute_overall_status()
            self.logger.info(f"Validation completed: {overall_status.value}")
            
            # Step 5: Create validation result
            result = ValidationResult(
                overall_status=overall_status,
                stage_results=self.stage_results,
                total_errors=self.error_collector.total_errors,
                total_warnings=self.error_collector.total_warnings,
                errors_by_category=self.error_collector.errors_by_category,
                errors_by_severity=self.error_collector.errors_by_severity,
                validation_metadata={
                    "total_files": len(file_paths),
                    "total_records": sum(len(rows) for rows in self.parsed_data.values()),
                    "execution_time": sum(sr.execution_time_seconds for sr in self.stage_results)
                }
            )
            
            # Step 6: Compute metrics
            self.logger.info("Computing quality metrics")
            quality_vector = self.quality_computer.compute(self.parsed_data, result)
            complexity_analysis = self.complexity_analyzer.analyze(self.stage_results)
            statistical_summary = self.statistical_summary.compute(self.parsed_data)
            
            # Step 7: Generate reports and outputs
            self.logger.info("Generating reports and metrics")
            self.metrics_writer.write_validation_metrics(result, {})
            self.metrics_writer.write_quality_metrics(quality_vector)
            self.metrics_writer.write_complexity_analysis(complexity_analysis)
            self.metrics_writer.write_statistical_summary(statistical_summary)
            
            # Generate comprehensive reports
            self.report_generator.generate(result)
            
            # Generate status report
            status_report = self.status_reporter.report(result)
            self.logger.info(f"Status Report: {status_report['summary']}")
            
            return result
            
        except Exception as e:
            self.logger.critical(f"Pipeline execution failed: {str(e)}")
            return self._create_failure_result(f"Pipeline error: {str(e)}")
        
        finally:
            self.logger.stop()
    
    def _parse_all_files(self, files: List[Path]) -> List[str]:
        """Parse all CSV files and store data."""
        errors = []
        
        for file_path in files:
            filename = file_path.name
            self.logger.debug(f"Parsing {filename}")
            
            parse_result = self.parser.parse_file(file_path)
            
            if not parse_result.success:
                errors.append(f"Failed to parse {filename}: {', '.join(parse_result.errors)}")
            else:
                self.parsed_data[filename] = parse_result.rows
                self.logger.debug(f"Parsed {filename}: {len(parse_result.rows)} rows")
        
        return errors
    
    def _execute_stage1(self, files: List[Path]) -> List:
        """Execute Stage 1: Syntactic Validation (per-file)."""
        results = []
        
        for file_path in files:
            filename = file_path.name
            result = self.stage1_validator.validate(file_path, filename)
            results.append(result)
        
        return results
    
    def _execute_stage2(self, files: List[Path]) -> List:
        """Execute Stage 2: Structural Validation (per-file)."""
        results = []
        
        for file_path in files:
            filename = file_path.name
            result = self.stage2_validator.validate(file_path, filename)
            results.append(result)
        
        return results
    
    def _collect_stage_errors(self, stage_results: List):
        """Collect errors from stage results."""
        for result in stage_results:
            self.error_collector.add_errors(result.errors)
            # Warnings are also errors with severity WARNING
            self.error_collector.add_errors(result.warnings)
    
    def _compute_overall_status(self) -> ValidationStatus:
        """Compute overall validation status from all stage results."""
        has_critical = any(
            ErrorSeverity.CRITICAL in self.error_collector.errors_by_severity
        )
        has_errors = self.error_collector.total_errors > 0
        has_warnings = self.error_collector.total_warnings > 0
        
        if has_critical or has_errors:
            return ValidationStatus.FAIL
        elif has_warnings:
            return ValidationStatus.WARNING
        else:
            return ValidationStatus.PASS
    
    def _create_failure_result(self, message: str, details: Optional[List[str]] = None) -> ValidationResult:
        """Create failure result with error message."""
        return ValidationResult(
            overall_status=ValidationStatus.FAIL,
            stage_results=self.stage_results,
            total_errors=len(details) if details else 1,
            total_warnings=0,
            validation_metadata={
                "error_message": message,
                "error_details": details or [message]
            }
        )

