"""
Progress Tracking System

Multi-bar progress tracking with Rich integration for professional console output.
Tracks overall progress, per-generator progress, validation progress, and system statistics.

Compliant with DESIGN_PART_5_CLI_AND_OUTPUT.md Section 5.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import time
import logging

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
)
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

logger = logging.getLogger(__name__)


@dataclass
class GenerationPhase:
    """Represents a generation phase with progress tracking."""
    
    name: str
    total_steps: int
    current_step: int = 0
    current_task: str = ""
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    entities_generated: int = 0
    errors: int = 0
    
    def start(self) -> None:
        """Start the phase timer."""
        self.start_time = time.time()
    
    def complete(self) -> None:
        """Complete the phase."""
        self.end_time = time.time()
        self.current_step = self.total_steps
    
    @property
    def duration(self) -> float:
        """Get phase duration in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time
    
    @property
    def is_complete(self) -> bool:
        """Check if phase is complete."""
        return self.current_step >= self.total_steps
    
    @property
    def progress_percentage(self) -> float:
        """Get progress percentage."""
        if self.total_steps == 0:
            return 0.0
        return (self.current_step / self.total_steps) * 100


@dataclass
class SystemStats:
    """System resource statistics."""
    
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_peak_mb: float = 0.0
    entities_generated: int = 0
    entities_per_second: float = 0.0
    start_time: Optional[float] = None
    
    def update(
        self,
        cpu: float,
        memory: float,
        entities: int,
    ) -> None:
        """Update statistics."""
        self.cpu_percent = cpu
        self.memory_mb = memory
        self.memory_peak_mb = max(self.memory_peak_mb, memory)
        self.entities_generated = entities
        
        # Calculate rate
        if self.start_time:
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                self.entities_per_second = entities / elapsed


class ProgressTracker:
    """
    Multi-bar progress tracker with system statistics.
    
    Provides:
    - Overall generation progress bar
    - Per-generator progress bars
    - Validation progress bar
    - Real-time system statistics panel
    - Color-coded output (success, warning, error, info)
    """
    
    def __init__(self, console: Optional[Console] = None, show_stats: bool = True):
        """
        Initialize progress tracker.
        
        Args:
            console: Rich console instance (creates new if None)
            show_stats: Whether to show system statistics panel
        """
        self.console = console or Console()
        self.show_stats = show_stats
        
        # Progress tracking
        self.phases: Dict[str, GenerationPhase] = {}
        self.current_phase: Optional[str] = None
        
        # System stats
        self.stats = SystemStats()
        
        # Rich components
        self.progress: Optional[Progress] = None
        self.live: Optional[Live] = None
        self.layout: Optional[Layout] = None
        
        # Task IDs
        self.task_ids: Dict[str, int] = {}
        
        # Overall tracking
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        logger.debug("ProgressTracker initialized")
    
    def initialize(self, phases: list[str], phase_steps: Dict[str, int]) -> None:
        """
        Initialize progress tracker with phases.
        
        Args:
            phases: List of phase names in order
            phase_steps: Dictionary mapping phase names to step counts
        """
        # Create phases
        for phase_name in phases:
            steps = phase_steps.get(phase_name, 1)
            self.phases[phase_name] = GenerationPhase(
                name=phase_name,
                total_steps=steps,
            )
        
        # Create progress bars
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=self.console,
            expand=True,
        )
        
        # Create layout
        self.layout = Layout()
        self.layout.split_column(
            Layout(name="progress", size=len(phases) + 3),
            Layout(name="stats", size=8) if self.show_stats else None,
        )
        
        # Start tracking
        self.start_time = time.time()
        self.stats.start_time = self.start_time
        
        logger.info(f"Progress tracker initialized with {len(phases)} phases")
    
    def start_phase(self, phase_name: str, description: Optional[str] = None) -> None:
        """
        Start a generation phase.
        
        Args:
            phase_name: Name of the phase to start
            description: Optional description for progress bar
        """
        if phase_name not in self.phases:
            logger.warning(f"Unknown phase: {phase_name}")
            return
        
        self.current_phase = phase_name
        phase = self.phases[phase_name]
        phase.start()
        
        # Add progress task
        if self.progress:
            desc = description or f"[cyan]{phase_name}[/cyan]"
            task_id = self.progress.add_task(desc, total=phase.total_steps)
            self.task_ids[phase_name] = task_id
        
        logger.info(f"Started phase: {phase_name}")
    
    def update_phase(
        self,
        phase_name: str,
        step: int,
        task: Optional[str] = None,
        entities: int = 0,
    ) -> None:
        """
        Update phase progress.
        
        Args:
            phase_name: Name of the phase
            step: Current step number
            task: Current task description
            entities: Number of entities generated
        """
        if phase_name not in self.phases:
            return
        
        phase = self.phases[phase_name]
        phase.current_step = step
        if task:
            phase.current_task = task
        phase.entities_generated += entities
        
        # Update progress bar
        if phase_name in self.task_ids and self.progress:
            self.progress.update(self.task_ids[phase_name], completed=step)
    
    def complete_phase(self, phase_name: str) -> None:
        """
        Complete a generation phase.
        
        Args:
            phase_name: Name of the phase to complete
        """
        if phase_name not in self.phases:
            return
        
        phase = self.phases[phase_name]
        phase.complete()
        
        # Update progress bar to 100%
        if phase_name in self.task_ids and self.progress:
            self.progress.update(
                self.task_ids[phase_name],
                completed=phase.total_steps,
            )
        
        logger.info(
            f"Completed phase: {phase_name} "
            f"({phase.duration:.2f}s, {phase.entities_generated} entities)"
        )
    
    def update_stats(self, cpu: float, memory: float, entities: int) -> None:
        """
        Update system statistics.
        
        Args:
            cpu: CPU usage percentage
            memory: Memory usage in MB
            entities: Total entities generated
        """
        self.stats.update(cpu, memory, entities)
    
    def render_layout(self) -> Layout:
        """Render the complete layout with progress bars and stats."""
        if not self.layout or not self.progress:
            return Layout()
        
        # Update progress section
        self.layout["progress"].update(Panel(self.progress, title="[bold]Generation Progress[/bold]", border_style="cyan"))
        
        # Update stats section
        if self.show_stats:
            stats_table = self._create_stats_table()
            self.layout["stats"].update(Panel(stats_table, title="[bold]System Statistics[/bold]", border_style="green"))
        
        return self.layout
    
    def _create_stats_table(self) -> Table:
        """Create statistics table."""
        table = Table.grid(padding=(0, 2))
        table.add_column(style="cyan", justify="right")
        table.add_column(style="white")
        
        # CPU
        cpu_style = "red" if self.stats.cpu_percent > 80 else "yellow" if self.stats.cpu_percent > 60 else "green"
        table.add_row("CPU:", f"[{cpu_style}]{self.stats.cpu_percent:.1f}%[/{cpu_style}]")
        
        # Memory
        mem_style = "red" if self.stats.memory_mb > 4000 else "yellow" if self.stats.memory_mb > 2000 else "green"
        table.add_row("Memory:", f"[{mem_style}]{self.stats.memory_mb:.1f} MB[/{mem_style}]")
        table.add_row("Peak Memory:", f"[dim]{self.stats.memory_peak_mb:.1f} MB[/dim]")
        
        # Entities
        table.add_row("Entities:", f"[yellow]{self.stats.entities_generated:,}[/yellow]")
        table.add_row("Rate:", f"[yellow]{self.stats.entities_per_second:.1f}/sec[/yellow]")
        
        # Time
        if self.start_time:
            elapsed = time.time() - self.start_time
            table.add_row("Elapsed:", f"[dim]{elapsed:.1f}s[/dim]")
        
        return table
    
    def start(self) -> None:
        """Start live display."""
        if self.layout:
            self.live = Live(
                self.render_layout(),
                console=self.console,
                refresh_per_second=4,
            )
            self.live.start()
    
    def stop(self) -> None:
        """Stop live display."""
        if self.live:
            self.live.stop()
            self.end_time = time.time()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get generation summary.
        
        Returns:
            Dictionary with summary statistics
        """
        total_duration = (self.end_time or time.time()) - (self.start_time or time.time())
        
        phase_summaries = {}
        for phase_name, phase in self.phases.items():
            phase_summaries[phase_name] = {
                "duration": phase.duration,
                "entities": phase.entities_generated,
                "errors": phase.errors,
                "completed": phase.is_complete,
            }
        
        return {
            "total_duration": total_duration,
            "total_entities": self.stats.entities_generated,
            "peak_memory_mb": self.stats.memory_peak_mb,
            "avg_rate": self.stats.entities_per_second,
            "phases": phase_summaries,
        }
    
    def print_summary(self) -> None:
        """Print generation summary."""
        summary = self.get_summary()
        
        table = Table(title="Generation Summary", show_header=True, header_style="bold cyan")
        table.add_column("Phase", style="cyan")
        table.add_column("Duration", justify="right", style="yellow")
        table.add_column("Entities", justify="right", style="green")
        table.add_column("Status", justify="center")
        
        for phase_name, phase_data in summary["phases"].items():
            status = "✓" if phase_data["completed"] else "✗"
            status_style = "green" if phase_data["completed"] else "red"
            
            table.add_row(
                phase_name,
                f"{phase_data['duration']:.2f}s",
                f"{phase_data['entities']:,}",
                f"[{status_style}]{status}[/{status_style}]",
            )
        
        # Add totals
        table.add_section()
        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold]{summary['total_duration']:.2f}s[/bold]",
            f"[bold]{summary['total_entities']:,}[/bold]",
            "",
        )
        
        self.console.print(table)
        
        # Additional stats
        self.console.print(f"\n[cyan]Average Rate:[/cyan] [yellow]{summary['avg_rate']:.1f} entities/sec[/yellow]")
        self.console.print(f"[cyan]Peak Memory:[/cyan] [yellow]{summary['peak_memory_mb']:.1f} MB[/yellow]")


class SimpleProgressTracker:
    """
    Simple progress tracker without Rich components.
    Used when --no-progress-bars flag is set.
    """
    
    def __init__(self, console: Optional[Console] = None):
        """Initialize simple tracker."""
        self.console = console or Console()
        self.phases: Dict[str, GenerationPhase] = {}
        self.start_time: Optional[float] = None
    
    def initialize(self, phases: list[str], phase_steps: Dict[str, int]) -> None:
        """Initialize phases."""
        for phase_name in phases:
            steps = phase_steps.get(phase_name, 1)
            self.phases[phase_name] = GenerationPhase(name=phase_name, total_steps=steps)
        self.start_time = time.time()
    
    def start_phase(self, phase_name: str, description: Optional[str] = None) -> None:
        """Start phase with simple log message."""
        if phase_name in self.phases:
            self.phases[phase_name].start()
            self.console.print(f"[cyan]Starting:[/cyan] {phase_name}")
    
    def update_phase(
        self,
        phase_name: str,
        step: int,
        task: Optional[str] = None,
        entities: int = 0,
    ) -> None:
        """Update phase (no-op for simple tracker)."""
        pass
    
    def complete_phase(self, phase_name: str) -> None:
        """Complete phase with simple log message."""
        if phase_name in self.phases:
            phase = self.phases[phase_name]
            phase.complete()
            self.console.print(
                f"[green]✓ Completed:[/green] {phase_name} ({phase.duration:.2f}s)"
            )
    
    def update_stats(self, cpu: float, memory: float, entities: int) -> None:
        """Update stats (no-op for simple tracker)."""
        pass
    
    def start(self) -> None:
        """Start (no-op for simple tracker)."""
        pass
    
    def stop(self) -> None:
        """Stop (no-op for simple tracker)."""
        pass
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary."""
        total_duration = time.time() - (self.start_time or time.time())
        total_entities = sum(p.entities_generated for p in self.phases.values())
        
        return {
            "total_duration": total_duration,
            "total_entities": total_entities,
            "phases": {
                name: {
                    "duration": phase.duration,
                    "entities": phase.entities_generated,
                    "completed": phase.is_complete,
                }
                for name, phase in self.phases.items()
            },
        }
    
    def print_summary(self) -> None:
        """Print simple summary."""
        summary = self.get_summary()
        self.console.print("\n[bold cyan]Generation Complete[/bold cyan]")
        self.console.print(f"Total Time: {summary['total_duration']:.2f}s")
        self.console.print(f"Total Entities: {summary['total_entities']:,}")
