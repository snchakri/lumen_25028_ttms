"""
Rich console logging system for real-time validation progress tracking.

Uses the rich library for beautiful, color-coded console output with
progress bars, status indicators, and formatted tables.
"""

from typing import Optional, Any, Dict
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TaskID
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich import box
from datetime import datetime


class ConsoleLogger:
    """
    Rich console logger for real-time validation progress.
    
    Provides:
    - Color-coded status messages (green=pass, red=fail, yellow=warning)
    - Progress bars for each validation stage
    - Real-time error display
    - Summary tables
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize console logger.
        
        Args:
            verbose: Enable verbose output (debug messages)
        """
        self.console = Console()
        self.verbose = verbose
        self.progress: Optional[Progress] = None
        self.current_task: Optional[TaskID] = None
        self.stage_tasks: Dict[int, TaskID] = {}
        self._start_time: Optional[datetime] = None
    
    def start_session(self, title: str = "Stage-1 Input Validation"):
        """Start validation session with header."""
        self._start_time = datetime.now()
        
        panel = Panel(
            f"[bold cyan]{title}[/bold cyan]\n"
            f"[dim]Started at: {self._start_time.strftime('%Y-%m-%d %H:%M:%S')}[/dim]\n"
            f"[dim]TEAM LUMEN [93912] - Theoretical Foundations Implementation[/dim]",
            box=box.DOUBLE,
            border_style="cyan"
        )
        self.console.print(panel)
        self.console.print()
    
    def end_session(self, status: str, total_errors: int, total_warnings: int):
        """End validation session with summary."""
        if self._start_time:
            duration = (datetime.now() - self._start_time).total_seconds()
            duration_str = f"{duration:.2f}s"
        else:
            duration_str = "unknown"
        
        # Status styling
        if status == "PASS":
            status_style = "bold green"
            status_icon = "[OK]"
        elif status == "WARNING":
            status_style = "bold yellow"
            status_icon = "[WARN]"
        else:
            status_style = "bold red"
            status_icon = "[X]"
        
        summary = (
            f"[{status_style}]{status_icon} Validation {status}[/{status_style}]\n"
            f"[dim]Duration: {duration_str}[/dim]\n"
            f"Errors: {total_errors}, Warnings: {total_warnings}"
        )
        
        panel = Panel(
            summary,
            box=box.DOUBLE,
            border_style="green" if status == "PASS" else ("yellow" if status == "WARNING" else "red")
        )
        self.console.print()
        self.console.print(panel)
    
    def start_progress(self, total_stages: int = 7):
        """Start progress tracking."""
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        )
        self.progress.start()
    
    def stop_progress(self):
        """Stop progress tracking."""
        if self.progress:
            self.progress.stop()
            self.progress = None
    
    def start_stage(self, stage_number: int, stage_name: str, total: int = 100):
        """Start a validation stage with progress bar."""
        if self.progress:
            task_id = self.progress.add_task(
                f"[cyan]Stage {stage_number}: {stage_name}[/cyan]",
                total=total
            )
            self.stage_tasks[stage_number] = task_id
            self.current_task = task_id
    
    def update_stage_progress(self, stage_number: int, advance: int = 1):
        """Update stage progress."""
        if self.progress and stage_number in self.stage_tasks:
            self.progress.update(self.stage_tasks[stage_number], advance=advance)
    
    def complete_stage(
        self,
        stage_number: int,
        stage_name: str,
        status: str,
        error_count: int = 0,
        warning_count: int = 0
    ):
        """Complete a validation stage."""
        if self.progress and stage_number in self.stage_tasks:
            task_id = self.stage_tasks[stage_number]
            self.progress.update(task_id, completed=100)
        
        # Status styling
        if status == "PASS":
            status_text = "[green][PASS][/green]"
        elif status == "WARNING":
            status_text = "[yellow][WARN][/yellow]"
        else:
            status_text = "[red][FAIL][/red]"
        
        message = f"Stage {stage_number}: {stage_name} - {status_text}"
        if error_count > 0:
            message += f" [red]({error_count} errors)[/red]"
        if warning_count > 0:
            message += f" [yellow]({warning_count} warnings)[/yellow]"
        
        self.console.print(message)
    
    def info(self, message: str):
        """Print info message."""
        self.console.print(f"[blue][INFO][/blue] {message}")
    
    def success(self, message: str):
        """Print success message."""
        self.console.print(f"[green][OK][/green] {message}")
    
    def warning(self, message: str):
        """Print warning message."""
        self.console.print(f"[yellow][WARN][/yellow] {message}")
    
    def error(self, message: str):
        """Print error message."""
        self.console.print(f"[red][FAIL][/red] {message}")
    
    def critical(self, message: str):
        """Print critical error message."""
        self.console.print(f"[bold red][CRITICAL][/bold red] {message}")
    
    def debug(self, message: str):
        """Print debug message (only if verbose)."""
        if self.verbose:
            self.console.print(f"[dim][DEBUG] {message}[/dim]")
    
    def print_file_list(self, title: str, files: list, style: str = "cyan"):
        """Print a formatted list of files."""
        if not files:
            return
        
        self.console.print(f"\n[bold {style}]{title}:[/bold {style}]")
        for file in files:
            self.console.print(f"  • {file}")
    
    def print_error_summary(self, errors: list, max_display: int = 10):
        """Print formatted error summary."""
        if not errors:
            return
        
        self.console.print(f"\n[bold red]Error Summary ({len(errors)} total):[/bold red]")
        
        # Group errors by category
        by_category = {}
        for error in errors:
            category = getattr(error, 'category', 'UNKNOWN')
            if hasattr(category, 'value'):
                category = category.value
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(error)
        
        # Display up to max_display errors
        displayed = 0
        for category, cat_errors in by_category.items():
            self.console.print(f"\n[yellow]{category}:[/yellow] {len(cat_errors)} errors")
            for error in cat_errors[:max_display - displayed]:
                self.console.print(f"  [red]•[/red] {error.message}")
                if error.file_path:
                    location = f"    [dim]{error.file_path}"
                    if error.line_number:
                        location += f", line {error.line_number}"
                    location += "[/dim]"
                    self.console.print(location)
                displayed += 1
                if displayed >= max_display:
                    break
            if displayed >= max_display:
                break
        
        if len(errors) > max_display:
            remaining = len(errors) - max_display
            self.console.print(f"\n[dim]... and {remaining} more errors[/dim]")
    
    def print_metrics_table(self, metrics: Dict[str, Any], title: str = "Validation Metrics"):
        """Print formatted metrics table."""
        table = Table(title=title, box=box.ROUNDED, show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="magenta")
        
        for key, value in metrics.items():
            # Format key (replace underscores with spaces, title case)
            formatted_key = key.replace('_', ' ').title()
            
            # Format value
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            elif isinstance(value, int):
                formatted_value = f"{value:,}"
            else:
                formatted_value = str(value)
            
            table.add_row(formatted_key, formatted_value)
        
        self.console.print()
        self.console.print(table)
    
    def print_quality_vector(self, quality: Dict[str, float]):
        """Print quality vector as a formatted display."""
        self.console.print("\n[bold cyan]Data Quality Vector:[/bold cyan]")
        
        for component, value in quality.items():
            # Determine color based on value
            if value >= 0.9:
                color = "green"
                icon = "●"
            elif value >= 0.7:
                color = "yellow"
                icon = "◐"
            else:
                color = "red"
                icon = "○"
            
            # Create progress bar
            bar_length = 20
            filled = int(value * bar_length)
            bar = "█" * filled + "░" * (bar_length - filled)
            
            formatted_component = component.replace('_', ' ').title()
            self.console.print(
                f"  {formatted_component:20s} [{color}]{icon}[/{color}] "
                f"[{color}]{bar}[/{color}] {value:.1%}"
            )
    
    def print_theorem_verification(self, theorem_id: str, verified: bool, details: str = ""):
        """Print theorem verification result."""
        if verified:
            status = "[green][OK] VERIFIED[/green]"
        else:
            status = "[red][X] FAILED[/red]"
        
        message = f"  {theorem_id}: {status}"
        if details:
            message += f" [dim]({details})[/dim]"
        
        self.console.print(message)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.progress:
            self.stop_progress()
        return False  # Don't suppress exceptions



        self.console.print(message)

    

    def info(self, message: str):

        """Print info message."""

        self.console.print(f"[blue][INFO][/blue] {message}")

    

    def success(self, message: str):

        """Print success message."""

        self.console.print(f"[green][OK][/green] {message}")

    

    def warning(self, message: str):

        """Print warning message."""

        self.console.print(f"[yellow][WARN][/yellow] {message}")

    

    def error(self, message: str):

        """Print error message."""

        self.console.print(f"[red][X][/red] {message}")

    

    def critical(self, message: str):

        """Print critical error message."""

        self.console.print(f"[bold red][WARN] CRITICAL:[/bold red] {message}")

    

    def debug(self, message: str):

        """Print debug message (only if verbose)."""

        if self.verbose:

            self.console.print(f"[dim][SCAN] {message}[/dim]")

    

    def print_file_list(self, title: str, files: list, style: str = "cyan"):

        """Print a formatted list of files."""

        if not files:

            return

        

        self.console.print(f"\n[bold {style}]{title}:[/bold {style}]")

        for file in files:

            self.console.print(f"  • {file}")

    

    def print_error_summary(self, errors: list, max_display: int = 10):

        """Print formatted error summary."""

        if not errors:

            return

        

        self.console.print(f"\n[bold red]Error Summary ({len(errors)} total):[/bold red]")

        

        # Group errors by category

        by_category = {}

        for error in errors:

            category = getattr(error, 'category', 'UNKNOWN')

            if hasattr(category, 'value'):

                category = category.value

            if category not in by_category:

                by_category[category] = []

            by_category[category].append(error)

        

        # Display up to max_display errors

        displayed = 0

        for category, cat_errors in by_category.items():

            self.console.print(f"\n[yellow]{category}:[/yellow] {len(cat_errors)} errors")

            for error in cat_errors[:max_display - displayed]:

                self.console.print(f"  [red]•[/red] {error.message}")

                if error.file_path:

                    location = f"    [dim]{error.file_path}"

                    if error.line_number:

                        location += f", line {error.line_number}"

                    location += "[/dim]"

                    self.console.print(location)

                displayed += 1

                if displayed >= max_display:

                    break

            if displayed >= max_display:

                break

        

        if len(errors) > max_display:

            remaining = len(errors) - max_display

            self.console.print(f"\n[dim]... and {remaining} more errors[/dim]")

    

    def print_metrics_table(self, metrics: Dict[str, Any], title: str = "Validation Metrics"):

        """Print formatted metrics table."""

        table = Table(title=title, box=box.ROUNDED, show_header=True, header_style="bold cyan")

        table.add_column("Metric", style="cyan", no_wrap=True)

        table.add_column("Value", justify="right", style="magenta")

        

        for key, value in metrics.items():

            # Format key (replace underscores with spaces, title case)

            formatted_key = key.replace('_', ' ').title()

            

            # Format value

            if isinstance(value, float):

                formatted_value = f"{value:.4f}"

            elif isinstance(value, int):

                formatted_value = f"{value:,}"

            else:

                formatted_value = str(value)

            

            table.add_row(formatted_key, formatted_value)

        

        self.console.print()

        self.console.print(table)

    

    def print_quality_vector(self, quality: Dict[str, float]):

        """Print quality vector as a formatted display."""

        self.console.print("\n[bold cyan]Data Quality Vector:[/bold cyan]")

        

        for component, value in quality.items():

            # Determine color based on value

            if value >= 0.9:

                color = "green"

                icon = "●"

            elif value >= 0.7:

                color = "yellow"

                icon = "◐"

            else:

                color = "red"

                icon = "○"

            

            # Create progress bar

            bar_length = 20

            filled = int(value * bar_length)

            bar = "█" * filled + "░" * (bar_length - filled)

            

            formatted_component = component.replace('_', ' ').title()

            self.console.print(

                f"  {formatted_component:20s} [{color}]{icon}[/{color}] "

                f"[{color}]{bar}[/{color}] {value:.1%}"

            )

    

    def print_theorem_verification(self, theorem_id: str, verified: bool, details: str = ""):

        """Print theorem verification result."""

        if verified:

            status = "[green][OK] VERIFIED[/green]"

        else:

            status = "[red][X] FAILED[/red]"

        

        message = f"  {theorem_id}: {status}"

        if details:

            message += f" [dim]({details})[/dim]"

        

        self.console.print(message)

    

    def __enter__(self):

        """Context manager entry."""

        return self

    

    def __exit__(self, exc_type, exc_val, exc_tb):

        """Context manager exit."""

        if self.progress:

            self.stop_progress()

        return False  # Don't suppress exceptions







        self.console.print(message)

    

    def info(self, message: str):

        """Print info message."""

        self.console.print(f"[blue][INFO][/blue] {message}")

    

    def success(self, message: str):

        """Print success message."""

        self.console.print(f"[green][OK][/green] {message}")

    

    def warning(self, message: str):

        """Print warning message."""

        self.console.print(f"[yellow][WARN][/yellow] {message}")

    

    def error(self, message: str):

        """Print error message."""

        self.console.print(f"[red][X][/red] {message}")

    

    def critical(self, message: str):

        """Print critical error message."""

        self.console.print(f"[bold red][WARN] CRITICAL:[/bold red] {message}")

    

    def debug(self, message: str):

        """Print debug message (only if verbose)."""

        if self.verbose:

            self.console.print(f"[dim][SCAN] {message}[/dim]")

    

    def print_file_list(self, title: str, files: list, style: str = "cyan"):

        """Print a formatted list of files."""

        if not files:

            return

        

        self.console.print(f"\n[bold {style}]{title}:[/bold {style}]")

        for file in files:

            self.console.print(f"  • {file}")

    

    def print_error_summary(self, errors: list, max_display: int = 10):

        """Print formatted error summary."""

        if not errors:

            return

        

        self.console.print(f"\n[bold red]Error Summary ({len(errors)} total):[/bold red]")

        

        # Group errors by category

        by_category = {}

        for error in errors:

            category = getattr(error, 'category', 'UNKNOWN')

            if hasattr(category, 'value'):

                category = category.value

            if category not in by_category:

                by_category[category] = []

            by_category[category].append(error)

        

        # Display up to max_display errors

        displayed = 0

        for category, cat_errors in by_category.items():

            self.console.print(f"\n[yellow]{category}:[/yellow] {len(cat_errors)} errors")

            for error in cat_errors[:max_display - displayed]:

                self.console.print(f"  [red]•[/red] {error.message}")

                if error.file_path:

                    location = f"    [dim]{error.file_path}"

                    if error.line_number:

                        location += f", line {error.line_number}"

                    location += "[/dim]"

                    self.console.print(location)

                displayed += 1

                if displayed >= max_display:

                    break

            if displayed >= max_display:

                break

        

        if len(errors) > max_display:

            remaining = len(errors) - max_display

            self.console.print(f"\n[dim]... and {remaining} more errors[/dim]")

    

    def print_metrics_table(self, metrics: Dict[str, Any], title: str = "Validation Metrics"):

        """Print formatted metrics table."""

        table = Table(title=title, box=box.ROUNDED, show_header=True, header_style="bold cyan")

        table.add_column("Metric", style="cyan", no_wrap=True)

        table.add_column("Value", justify="right", style="magenta")

        

        for key, value in metrics.items():

            # Format key (replace underscores with spaces, title case)

            formatted_key = key.replace('_', ' ').title()

            

            # Format value

            if isinstance(value, float):

                formatted_value = f"{value:.4f}"

            elif isinstance(value, int):

                formatted_value = f"{value:,}"

            else:

                formatted_value = str(value)

            

            table.add_row(formatted_key, formatted_value)

        

        self.console.print()

        self.console.print(table)

    

    def print_quality_vector(self, quality: Dict[str, float]):

        """Print quality vector as a formatted display."""

        self.console.print("\n[bold cyan]Data Quality Vector:[/bold cyan]")

        

        for component, value in quality.items():

            # Determine color based on value

            if value >= 0.9:

                color = "green"

                icon = "●"

            elif value >= 0.7:

                color = "yellow"

                icon = "◐"

            else:

                color = "red"

                icon = "○"

            

            # Create progress bar

            bar_length = 20

            filled = int(value * bar_length)

            bar = "█" * filled + "░" * (bar_length - filled)

            

            formatted_component = component.replace('_', ' ').title()

            self.console.print(

                f"  {formatted_component:20s} [{color}]{icon}[/{color}] "

                f"[{color}]{bar}[/{color}] {value:.1%}"

            )

    

    def print_theorem_verification(self, theorem_id: str, verified: bool, details: str = ""):

        """Print theorem verification result."""

        if verified:

            status = "[green][OK] VERIFIED[/green]"

        else:

            status = "[red][X] FAILED[/red]"

        

        message = f"  {theorem_id}: {status}"

        if details:

            message += f" [dim]({details})[/dim]"

        

        self.console.print(message)

    

    def __enter__(self):

        """Context manager entry."""

        return self

    

    def __exit__(self, exc_type, exc_val, exc_tb):

        """Context manager exit."""

        if self.progress:

            self.stop_progress()

        return False  # Don't suppress exceptions







        self.console.print(message)

    

    def info(self, message: str):

        """Print info message."""

        self.console.print(f"[blue][INFO][/blue] {message}")

    

    def success(self, message: str):

        """Print success message."""

        self.console.print(f"[green][OK][/green] {message}")

    

    def warning(self, message: str):

        """Print warning message."""

        self.console.print(f"[yellow][WARN][/yellow] {message}")

    

    def error(self, message: str):

        """Print error message."""

        self.console.print(f"[red][X][/red] {message}")

    

    def critical(self, message: str):

        """Print critical error message."""

        self.console.print(f"[bold red][WARN] CRITICAL:[/bold red] {message}")

    

    def debug(self, message: str):

        """Print debug message (only if verbose)."""

        if self.verbose:

            self.console.print(f"[dim][SCAN] {message}[/dim]")

    

    def print_file_list(self, title: str, files: list, style: str = "cyan"):

        """Print a formatted list of files."""

        if not files:

            return

        

        self.console.print(f"\n[bold {style}]{title}:[/bold {style}]")

        for file in files:

            self.console.print(f"  • {file}")

    

    def print_error_summary(self, errors: list, max_display: int = 10):

        """Print formatted error summary."""

        if not errors:

            return

        

        self.console.print(f"\n[bold red]Error Summary ({len(errors)} total):[/bold red]")

        

        # Group errors by category

        by_category = {}

        for error in errors:

            category = getattr(error, 'category', 'UNKNOWN')

            if hasattr(category, 'value'):

                category = category.value

            if category not in by_category:

                by_category[category] = []

            by_category[category].append(error)

        

        # Display up to max_display errors

        displayed = 0

        for category, cat_errors in by_category.items():

            self.console.print(f"\n[yellow]{category}:[/yellow] {len(cat_errors)} errors")

            for error in cat_errors[:max_display - displayed]:

                self.console.print(f"  [red]•[/red] {error.message}")

                if error.file_path:

                    location = f"    [dim]{error.file_path}"

                    if error.line_number:

                        location += f", line {error.line_number}"

                    location += "[/dim]"

                    self.console.print(location)

                displayed += 1

                if displayed >= max_display:

                    break

            if displayed >= max_display:

                break

        

        if len(errors) > max_display:

            remaining = len(errors) - max_display

            self.console.print(f"\n[dim]... and {remaining} more errors[/dim]")

    

    def print_metrics_table(self, metrics: Dict[str, Any], title: str = "Validation Metrics"):

        """Print formatted metrics table."""

        table = Table(title=title, box=box.ROUNDED, show_header=True, header_style="bold cyan")

        table.add_column("Metric", style="cyan", no_wrap=True)

        table.add_column("Value", justify="right", style="magenta")

        

        for key, value in metrics.items():

            # Format key (replace underscores with spaces, title case)

            formatted_key = key.replace('_', ' ').title()

            

            # Format value

            if isinstance(value, float):

                formatted_value = f"{value:.4f}"

            elif isinstance(value, int):

                formatted_value = f"{value:,}"

            else:

                formatted_value = str(value)

            

            table.add_row(formatted_key, formatted_value)

        

        self.console.print()

        self.console.print(table)

    

    def print_quality_vector(self, quality: Dict[str, float]):

        """Print quality vector as a formatted display."""

        self.console.print("\n[bold cyan]Data Quality Vector:[/bold cyan]")

        

        for component, value in quality.items():

            # Determine color based on value

            if value >= 0.9:

                color = "green"

                icon = "●"

            elif value >= 0.7:

                color = "yellow"

                icon = "◐"

            else:

                color = "red"

                icon = "○"

            

            # Create progress bar

            bar_length = 20

            filled = int(value * bar_length)

            bar = "█" * filled + "░" * (bar_length - filled)

            

            formatted_component = component.replace('_', ' ').title()

            self.console.print(

                f"  {formatted_component:20s} [{color}]{icon}[/{color}] "

                f"[{color}]{bar}[/{color}] {value:.1%}"

            )

    

    def print_theorem_verification(self, theorem_id: str, verified: bool, details: str = ""):

        """Print theorem verification result."""

        if verified:

            status = "[green][OK] VERIFIED[/green]"

        else:

            status = "[red][X] FAILED[/red]"

        

        message = f"  {theorem_id}: {status}"

        if details:

            message += f" [dim]({details})[/dim]"

        

        self.console.print(message)

    

    def __enter__(self):

        """Context manager entry."""

        return self

    

    def __exit__(self, exc_type, exc_val, exc_tb):

        """Context manager exit."""

        if self.progress:

            self.stop_progress()

        return False  # Don't suppress exceptions






