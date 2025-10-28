"""
Validation Orchestrator

Coordinates execution of all validation layers (L1-L7) with different modes
and generates comprehensive validation reports.

See DESIGN_PART_4_VALIDATION_FRAMEWORK.md Section 9.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.validation.validation_context import ValidationContext
from src.validation.error_models import ValidationReport, ErrorSeverity
from src.validation.layers import (
    L1StructuralValidator,
    L2DomainValidator,
    L3TemporalValidator,
    L4RelationalValidator,
    L5BusinessValidator,
    L6LtreeValidator,
    L7SchedulingValidator,
)


class ValidationOrchestrator:
    """
    Orchestrates execution of all validation layers.
    
    Modes:
        - strict: All layers must pass (default)
        - lenient: Only L1, L3 must pass
        - adversarial: All layers run but don't block
    
    Attributes:
        context: Validation context
        mode: Validation mode
        console: Rich console for output
    """
    
    def __init__(
        self,
        context: ValidationContext,
        mode: str = "strict",
        verbose: bool = True
    ):
        """
        Initialize orchestrator.
        
        Args:
            context: Validation context with state and config
            mode: Validation mode (strict/lenient/adversarial)
            verbose: Whether to show progress output
        """
        self.context = context
        self.mode = mode
        self.verbose = verbose
        self.console = Console() if verbose else None
        
        # Initialize validators
        self.validators = [
            L1StructuralValidator(context),
            L2DomainValidator(context),
            L3TemporalValidator(context),
            L4RelationalValidator(context),
            L5BusinessValidator(context),
            L6LtreeValidator(context),
            L7SchedulingValidator(context),
        ]
    
    def validate_all(self) -> ValidationReport:
        """
        Execute all validation layers and generate report.
        
        Returns:
            ValidationReport with results from all layers
        """
        report = ValidationReport(mode=self.mode)
        
        if self.verbose and self.console:
            self.console.print(Panel.fit(
                f"[bold cyan]Validation Started[/bold cyan]\n"
                f"Mode: {self.mode.upper()}\n"
                f"Layers: L1-L7",
                title="Validation Orchestrator"
            ))
        
        # Execute each layer
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            disable=not self.verbose
        ) as progress:
            
            for validator in self.validators:
                layer_name = validator.get_layer_name()
                task = progress.add_task(f"[cyan]{layer_name}...", total=None)
                
                # Run validation
                result = validator.validate_all_entities()
                report.add_layer_result(result)
                
                # Display result
                status = "✓ PASSED" if result.passed else "✗ FAILED"
                color = "green" if result.passed else "red"
                progress.update(task, description=f"[{color}]{layer_name}: {status}")
                
                # In strict mode, stop on critical errors
                if self.mode == "strict" and result.get_critical_count() > 0:
                    if self.console:
                        self.console.print(f"\n[bold red]CRITICAL ERRORS in {layer_name}[/bold red]")
                        self.console.print(f"Stopping validation due to critical errors.\n")
                    break
                
                # In strict mode, L1 and L3 must pass
                if self.mode == "strict" and layer_name in ["L1_Structural", "L3_Temporal"]:
                    if not result.passed:
                        if self.console:
                            self.console.print(f"\n[bold red]{layer_name} FAILED[/bold red]")
                            self.console.print(f"Cannot continue without passing {layer_name}.\n")
                        break
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)
        
        # Display summary
        if self.verbose and self.console:
            self._display_summary(report)
        
        return report
    
    def _generate_recommendations(self, report: ValidationReport) -> List[str]:
        """Generate fix recommendations based on errors."""
        recommendations = []
        
        all_errors = report.get_all_errors()
        
        # Count by constraint type
        constraint_counts: Dict[str, int] = {}
        for error in all_errors:
            constraint = error.constraint_name or "UNKNOWN"
            constraint_counts[constraint] = constraint_counts.get(constraint, 0) + 1
        
        # Top issues
        top_constraints = sorted(constraint_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for constraint, count in top_constraints:
            if constraint == "UUID_FORMAT":
                recommendations.append(
                    f"Fix {count} UUID format errors: Ensure all IDs use uuid4() or uuid7()"
                )
            elif constraint == "NOT_NULL":
                recommendations.append(
                    f"Fix {count} null value errors: Verify all required fields are populated"
                )
            elif constraint == "FOREIGN_KEY_INTEGRITY":
                recommendations.append(
                    f"Fix {count} foreign key errors: Ensure referenced entities exist before creating dependents"
                )
            elif constraint == "TIME_ORDERING":
                recommendations.append(
                    f"Fix {count} time ordering errors: Ensure start_time < end_time"
                )
            elif constraint == "CREDIT_ABSOLUTE_LIMIT":
                recommendations.append(
                    f"Fix {count} credit limit violations: Reduce student course loads to ≤27 credits"
                )
            else:
                recommendations.append(
                    f"Fix {count} {constraint} errors: Review constraint definition and generation logic"
                )
        
        if not recommendations:
            recommendations.append("No errors detected. All validations passed!")
        
        return recommendations
    
    def _display_summary(self, report: ValidationReport) -> None:
        """Display validation summary with Rich formatting."""
        if not self.console:
            return
        
        # Summary table
        table = Table(title="Validation Summary", show_header=True, header_style="bold magenta")
        table.add_column("Layer", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Entities", justify="right")
        table.add_column("Critical", justify="right", style="red")
        table.add_column("Errors", justify="right", style="yellow")
        table.add_column("Warnings", justify="right", style="blue")
        table.add_column("Time", justify="right")
        
        for result in report.layer_results:
            status = "✓" if result.passed else "✗"
            status_color = "green" if result.passed else "red"
            
            table.add_row(
                result.layer_name,
                f"[{status_color}]{status}[/{status_color}]",
                str(result.entities_validated),
                str(result.get_critical_count()),
                str(result.get_error_count()),
                str(result.get_warning_count()),
                f"{result.execution_time_seconds:.2f}s"
            )
        
        self.console.print("\n")
        self.console.print(table)
        
        # Overall status
        status_text = "VALIDATION PASSED ✓" if report.overall_passed else "VALIDATION FAILED ✗"
        status_color = "green" if report.overall_passed else "red"
        
        self.console.print("\n")
        self.console.print(Panel.fit(
            f"[bold {status_color}]{status_text}[/bold {status_color}]\n\n"
            f"Total Entities: {report.total_entities}\n"
            f"Total Violations: {report.total_violations}\n"
            f"Total Time: {report.total_time_seconds:.2f}s",
            title="Overall Result"
        ))
        
        # Recommendations
        if report.recommendations:
            self.console.print("\n[bold cyan]Recommendations:[/bold cyan]")
            for i, rec in enumerate(report.recommendations, 1):
                self.console.print(f"  {i}. {rec}")
        
        self.console.print("\n")
    
    def save_report(self, report: ValidationReport, output_dir: Path) -> None:
        """
        Save validation report to JSON and text files.
        
        Args:
            report: Validation report to save
            output_dir: Directory for output files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON report
        json_file = output_dir / f"validation_report_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        
        # Text summary
        text_file = output_dir / f"validation_summary_{timestamp}.txt"
        with open(text_file, 'w') as f:
            f.write(report.get_summary())
        
        if self.verbose and self.console:
            self.console.print(f"\n[green]Reports saved:[/green]")
            self.console.print(f"  JSON: {json_file}")
            self.console.print(f"  Text: {text_file}")
