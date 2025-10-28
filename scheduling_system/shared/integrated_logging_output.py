"""
Integrated Logging and Output System
Unified system that integrates with existing output structure and provides
comprehensive logging and output management for all 7 stages.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone
import structlog

from .logging_system import StructuredLogger, LogCategory, create_logging_system
from .output_manager import OutputManager, OutputType, OutputFormat, create_output_manager

logger = structlog.get_logger(__name__)

class IntegratedLoggingOutputSystem:
    """
    Integrated system that combines logging and output management
    with existing test system and stage-specific output structures.
    """
    
    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or Path("scheduling_engine_localized")
        
        # Initialize logging system
        self.logging_system = create_logging_system(self.base_dir / "logs")
        self.session_logger = self.logging_system.start_session()
        
        # Initialize output manager with existing structure integration
        self.output_manager = create_output_manager(
            base_output_dir=self.base_dir / "outputs",
            integrate_with_existing_structure=True
        )
        
        # Track existing output directories
        self.existing_output_dirs = {
            "test_results": self.base_dir / "test_system" / "test_results",
            "stage3_outputs": self.base_dir / "stage3_outputs",
            "stage5_outputs": self.base_dir / "stage_5_2_output",
            "stage7_logs": self.base_dir / "logs"
        }
        
        # Initialize logger first
        self.logger = logger.bind(
            session_id=self.session_logger.session_id,
            execution_id=self.session_logger.execution_id
        )
        
        # Verify existing structure
        self._verify_existing_structure()
        
        self.logger.info("Integrated logging and output system initialized")
    
    def _verify_existing_structure(self):
        """Verify and document existing output structure."""
        self.logger.info("Verifying existing output structure")
        
        for name, path in self.existing_output_dirs.items():
            if path.exists():
                self.logger.info(f"Found existing output directory: {name} at {path}")
                
                # Count existing outputs
                if name == "test_results":
                    test_cases = list(path.glob("tc_*"))
                    self.logger.info(f"Found {len(test_cases)} test cases in test_results")
                    
                    # Log test case details
                    for test_case in test_cases[:3]:  # Log first 3 as examples
                        self.logger.info(f"Test case: {test_case.name}")
                else:
                    files = list(path.glob("*"))
                    self.logger.info(f"Found {len(files)} files in {name}")
            else:
                self.logger.info(f"Output directory not found (will be created): {name} at {path}")
    
    def log_stage_execution(self, stage: int, stage_name: str, 
                          start_time: float = None, **context):
        """Log stage execution start with existing structure awareness."""
        if start_time is None:
            start_time = time.time()
        
        # Log to structured logger
        self.session_logger.log_stage_start(stage, stage_name, **context)
        
        # Log existing outputs for this stage
        existing_outputs = self._get_existing_stage_outputs(stage)
        if existing_outputs:
            self.logger.info(f"Found existing outputs for stage {stage}", 
                           existing_outputs=existing_outputs)
        
        return start_time
    
    def log_stage_completion(self, stage: int, stage_name: str,
                           success: bool, execution_time: float = None,
                           outputs_created: List[str] = None, **metrics):
        """Log stage completion with output tracking."""
        if execution_time is None:
            execution_time = time.time() - self.session_logger.performance_tracker.get(
                f"stage_{stage}_start", time.time()
            )
        
        # Add output information to metrics
        if outputs_created:
            metrics["outputs_created"] = outputs_created
            metrics["output_count"] = len(outputs_created)
        
        # Log to structured logger
        self.session_logger.log_stage_completion(stage, stage_name, success, **metrics)
        
        # Log to integrated logger
        self.logger.info(f"Stage {stage} ({stage_name}) completed",
                        success=success,
                        execution_time=execution_time,
                        outputs_created=outputs_created or [])
    
    def save_stage_output(self, stage: int, component: str, data: Any,
                         output_type: OutputType, format: OutputFormat = OutputFormat.CSV,
                         save_to_existing_structure: bool = True, **metadata):
        """
        Save stage output with integration to existing structure.
        
        Args:
            stage: Stage number (1-7)
            component: Component name
            data: Data to save
            output_type: Type of output
            format: Output format
            save_to_existing_structure: Whether to save to existing directory structure
            **metadata: Additional metadata
        """
        self.logger.info(f"Saving output for stage {stage}, component {component}",
                        output_type=output_type.value,
                        format=format.value)
        
        # Save to new output manager
        file_path = self.output_manager.save_stage_output(
            stage, component, data, output_type, format, metadata
        )
        
        # Also save to existing structure if requested
        if save_to_existing_structure:
            existing_path = self._save_to_existing_structure(
                stage, component, data, format
            )
            if existing_path:
                self.logger.info(f"Also saved to existing structure: {existing_path}")
        
        return file_path
    
    def _get_existing_stage_outputs(self, stage: int) -> List[str]:
        """Get list of existing outputs for a stage."""
        existing_outputs = []
        
        # Check test results
        test_results_dir = self.existing_output_dirs["test_results"]
        if test_results_dir.exists():
            test_cases = list(test_results_dir.glob("tc_*"))
            existing_outputs.extend([f"test_case_{tc.name}" for tc in test_cases])
        
        # Check stage-specific directories
        if stage == 3:
            stage3_dir = self.existing_output_dirs["stage3_outputs"]
            if stage3_dir.exists():
                files = list(stage3_dir.glob("*"))
                existing_outputs.extend([f"stage3_{f.name}" for f in files])
        elif stage == 5:
            stage5_dir = self.existing_output_dirs["stage5_outputs"]
            if stage5_dir.exists():
                files = list(stage5_dir.glob("*"))
                existing_outputs.extend([f"stage5_{f.name}" for f in files])
        
        return existing_outputs
    
    def _save_to_existing_structure(self, stage: int, component: str, 
                                   data: Any, format: OutputFormat) -> Optional[str]:
        """Save to existing directory structure."""
        try:
            if stage <= 2:  # Stages 1-2: Save to test_results structure
                return self._save_to_test_results(stage, component, data, format)
            elif stage == 3:  # Stage 3: Save to stage3_outputs
                return self._save_to_stage3_outputs(component, data, format)
            elif stage == 5:  # Stage 5: Save to stage5_outputs
                return self._save_to_stage5_outputs(component, data, format)
            else:  # Other stages: Save to general outputs
                return self._save_to_general_outputs(stage, component, data, format)
        except Exception as e:
            self.logger.warning(f"Failed to save to existing structure: {e}")
            return None
    
    def _save_to_test_results(self, stage: int, component: str, 
                             data: Any, format: OutputFormat) -> Optional[str]:
        """Save to test results structure."""
        test_results_dir = self.existing_output_dirs["test_results"]
        if not test_results_dir.exists():
            return None
        
        # Create a new test case directory
        timestamp = int(time.time())
        test_case_dir = test_results_dir / f"tc_{timestamp}_{component}"
        test_case_dir.mkdir(exist_ok=True)
        
        # Save data
        file_path = test_case_dir / f"{component}.{format.value}"
        self._write_data_to_file(data, file_path, format)
        
        # Save metadata
        metadata = {
            "stage": stage,
            "component": component,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "format": format.value,
            "session_id": self.session_logger.session_id
        }
        
        metadata_path = test_case_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return str(file_path)
    
    def _save_to_stage3_outputs(self, component: str, data: Any, format: OutputFormat) -> Optional[str]:
        """Save to stage 3 outputs."""
        stage3_dir = self.existing_output_dirs["stage3_outputs"]
        stage3_dir.mkdir(exist_ok=True)
        
        file_path = stage3_dir / f"{component}.{format.value}"
        self._write_data_to_file(data, file_path, format)
        return str(file_path)
    
    def _save_to_stage5_outputs(self, component: str, data: Any, format: OutputFormat) -> Optional[str]:
        """Save to stage 5 outputs."""
        stage5_dir = self.existing_output_dirs["stage5_outputs"]
        stage5_dir.mkdir(exist_ok=True)
        
        file_path = stage5_dir / f"{component}.{format.value}"
        self._write_data_to_file(data, file_path, format)
        return str(file_path)
    
    def _save_to_general_outputs(self, stage: int, component: str, 
                                data: Any, format: OutputFormat) -> Optional[str]:
        """Save to general outputs directory."""
        outputs_dir = self.base_dir / "outputs"
        stage_dir = outputs_dir / f"stage_{stage}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = stage_dir / f"{component}.{format.value}"
        self._write_data_to_file(data, file_path, format)
        return str(file_path)
    
    def _write_data_to_file(self, data: Any, file_path: Path, format: OutputFormat):
        """Write data to file based on format."""
        if format == OutputFormat.CSV:
            if hasattr(data, 'to_csv'):
                data.to_csv(file_path, index=False)
            else:
                import pandas as pd
                pd.DataFrame(data if isinstance(data, list) else [data]).to_csv(file_path, index=False)
        elif format == OutputFormat.JSON:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format for existing structure: {format}")
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive report including existing outputs."""
        self.logger.info("Generating comprehensive execution report")
        
        # Get reports from both systems
        logging_report = self.session_logger.generate_session_report()
        output_report = self.output_manager.generate_execution_report()
        
        # Analyze existing outputs
        existing_outputs_analysis = self._analyze_existing_outputs()
        
        # Combine reports
        comprehensive_report = {
            "session_id": self.session_logger.session_id,
            "execution_id": self.session_logger.execution_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "logging_report": logging_report,
            "output_report": output_report,
            "existing_outputs_analysis": existing_outputs_analysis,
            "system_integration": {
                "existing_structure_integrated": True,
                "test_results_available": self.existing_output_dirs["test_results"].exists(),
                "stage_specific_outputs_available": any(
                    path.exists() for name, path in self.existing_output_dirs.items() 
                    if name != "test_results"
                )
            }
        }
        
        # Save comprehensive report
        report_path = self.base_dir / "comprehensive_execution_report.json"
        with open(report_path, 'w') as f:
            json.dump(comprehensive_report, f, indent=2)
        
        self.logger.info(f"Comprehensive report saved to {report_path}")
        return comprehensive_report
    
    def _analyze_existing_outputs(self) -> Dict[str, Any]:
        """Analyze existing output structure."""
        analysis = {
            "test_results": {"available": False, "test_cases": 0, "total_files": 0},
            "stage_outputs": {},
            "total_existing_outputs": 0
        }
        
        # Analyze test results
        test_results_dir = self.existing_output_dirs["test_results"]
        if test_results_dir.exists():
            test_cases = list(test_results_dir.glob("tc_*"))
            total_files = sum(len(list(tc.glob("*"))) for tc in test_cases)
            
            analysis["test_results"] = {
                "available": True,
                "test_cases": len(test_cases),
                "total_files": total_files,
                "test_case_details": [tc.name for tc in test_cases[:5]]  # First 5 as examples
            }
            analysis["total_existing_outputs"] += total_files
        
        # Analyze stage-specific outputs
        for name, path in self.existing_output_dirs.items():
            if name != "test_results" and path.exists():
                files = list(path.glob("*"))
                analysis["stage_outputs"][name] = {
                    "available": True,
                    "file_count": len(files),
                    "files": [f.name for f in files[:5]]  # First 5 as examples
                }
                analysis["total_existing_outputs"] += len(files)
        
        return analysis

# Factory function
def create_integrated_logging_output_system(base_dir: Path = None) -> IntegratedLoggingOutputSystem:
    """
    Create integrated logging and output system.
    
    Args:
        base_dir: Base directory for the scheduling engine
        
    Returns:
        Configured IntegratedLoggingOutputSystem
    """
    return IntegratedLoggingOutputSystem(base_dir)

