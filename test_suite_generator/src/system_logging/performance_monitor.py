"""
Performance Monitoring System

Tracks CPU, memory, disk I/O, and generation rate metrics.
Provides performance warnings and statistics.

Compliant with DESIGN_PART_5_CLI_AND_OUTPUT.md Section 6.
"""

import time
import psutil
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class PerformanceSnapshot:
    """Single point-in-time performance measurement."""
    
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_read_mb: float
    disk_write_mb: float
    entities_generated: int
    generation_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": datetime.fromtimestamp(self.timestamp).isoformat(),
            "cpu_percent": round(self.cpu_percent, 2),
            "memory_mb": round(self.memory_mb, 2),
            "memory_percent": round(self.memory_percent, 2),
            "disk_read_mb": round(self.disk_read_mb, 2),
            "disk_write_mb": round(self.disk_write_mb, 2),
            "entities_generated": self.entities_generated,
            "generation_rate": round(self.generation_rate, 2),
        }


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics."""
    
    start_time: float
    end_time: Optional[float] = None
    
    # CPU metrics
    cpu_min: float = 100.0
    cpu_max: float = 0.0
    cpu_avg: float = 0.0
    cpu_samples: List[float] = field(default_factory=list)
    
    # Memory metrics
    memory_min_mb: float = float('inf')
    memory_max_mb: float = 0.0
    memory_avg_mb: float = 0.0
    memory_peak_mb: float = 0.0
    memory_samples: List[float] = field(default_factory=list)
    
    # Disk I/O metrics
    disk_read_total_mb: float = 0.0
    disk_write_total_mb: float = 0.0
    
    # Generation metrics
    total_entities: int = 0
    generation_rate_avg: float = 0.0
    generation_rate_peak: float = 0.0
    
    # Warnings
    warnings: List[str] = field(default_factory=list)
    
    def add_cpu_sample(self, value: float) -> None:
        """Add CPU sample and update metrics."""
        self.cpu_samples.append(value)
        self.cpu_min = min(self.cpu_min, value)
        self.cpu_max = max(self.cpu_max, value)
        self.cpu_avg = sum(self.cpu_samples) / len(self.cpu_samples)
        
        # Check for warnings
        if value > 90:
            self.warnings.append(f"High CPU usage: {value:.1f}%")
    
    def add_memory_sample(self, value_mb: float) -> None:
        """Add memory sample and update metrics."""
        self.memory_samples.append(value_mb)
        self.memory_min_mb = min(self.memory_min_mb, value_mb)
        self.memory_max_mb = max(self.memory_max_mb, value_mb)
        self.memory_peak_mb = max(self.memory_peak_mb, value_mb)
        self.memory_avg_mb = sum(self.memory_samples) / len(self.memory_samples)
        
        # Check for warnings
        if value_mb > 4000:
            self.warnings.append(f"High memory usage: {value_mb:.1f} MB")
    
    def update_generation(self, entities: int, rate: float) -> None:
        """Update generation metrics."""
        self.total_entities = entities
        self.generation_rate_peak = max(self.generation_rate_peak, rate)
        
        # Calculate average rate
        if self.end_time:
            duration = self.end_time - self.start_time
        else:
            duration = time.time() - self.start_time
        
        if duration > 0:
            self.generation_rate_avg = entities / duration
    
    def finalize(self) -> None:
        """Finalize metrics at end of run."""
        self.end_time = time.time()
    
    @property
    def duration_seconds(self) -> float:
        """Get total duration in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "duration_seconds": round(self.duration_seconds, 2),
            "cpu": {
                "min_percent": round(self.cpu_min, 2),
                "max_percent": round(self.cpu_max, 2),
                "avg_percent": round(self.cpu_avg, 2),
            },
            "memory": {
                "min_mb": round(self.memory_min_mb, 2),
                "max_mb": round(self.memory_max_mb, 2),
                "peak_mb": round(self.memory_peak_mb, 2),
                "avg_mb": round(self.memory_avg_mb, 2),
            },
            "disk_io": {
                "read_total_mb": round(self.disk_read_total_mb, 2),
                "write_total_mb": round(self.disk_write_total_mb, 2),
            },
            "generation": {
                "total_entities": self.total_entities,
                "avg_rate": round(self.generation_rate_avg, 2),
                "peak_rate": round(self.generation_rate_peak, 2),
            },
            "warnings": self.warnings,
        }


class PerformanceMonitor:
    """
    Performance monitoring system.
    
    Tracks:
    - CPU usage (overall and per-core)
    - Memory usage (RSS, virtual, peak)
    - Disk I/O (read/write bytes)
    - Generation rate (entities per second)
    
    Provides:
    - Real-time metrics
    - Performance warnings
    - Historical snapshots
    - Summary statistics
    """
    
    def __init__(self, enabled: bool = True, sample_interval: float = 1.0):
        """
        Initialize performance monitor.
        
        Args:
            enabled: Whether monitoring is enabled
            sample_interval: Seconds between samples
        """
        self.enabled = enabled
        self.sample_interval = sample_interval
        
        # Process info
        self.process = psutil.Process()
        self.start_time = time.time()
        
        # Metrics
        self.metrics = PerformanceMetrics(start_time=self.start_time)
        
        # Snapshots
        self.snapshots: List[PerformanceSnapshot] = []
        self.last_snapshot_time = self.start_time
        
        # Disk I/O baseline
        self.disk_io_start = psutil.disk_io_counters()
        
        # Current entities count
        self.current_entities = 0
        
        logger.debug(f"PerformanceMonitor initialized (enabled={enabled})")
    
    def should_sample(self) -> bool:
        """Check if enough time has passed for next sample."""
        if not self.enabled:
            return False
        return (time.time() - self.last_snapshot_time) >= self.sample_interval
    
    def update_entities(self, count: int) -> None:
        """
        Update entity count.
        
        Args:
            count: New total entity count
        """
        self.current_entities = count
    
    def take_snapshot(self) -> Optional[PerformanceSnapshot]:
        """
        Take performance snapshot.
        
        Returns:
            PerformanceSnapshot if monitoring enabled, None otherwise
        """
        if not self.enabled:
            return None
        
        try:
            # Get current metrics
            cpu_percent = self.process.cpu_percent(interval=0.1)
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            memory_percent = self.process.memory_percent()
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io and self.disk_io_start:
                disk_read_mb = (disk_io.read_bytes - self.disk_io_start.read_bytes) / (1024 * 1024)
                disk_write_mb = (disk_io.write_bytes - self.disk_io_start.write_bytes) / (1024 * 1024)
            else:
                disk_read_mb = 0.0
                disk_write_mb = 0.0
            
            # Generation rate
            elapsed = time.time() - self.start_time
            generation_rate = self.current_entities / elapsed if elapsed > 0 else 0.0
            
            # Create snapshot
            snapshot = PerformanceSnapshot(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                memory_percent=memory_percent,
                disk_read_mb=disk_read_mb,
                disk_write_mb=disk_write_mb,
                entities_generated=self.current_entities,
                generation_rate=generation_rate,
            )
            
            # Update metrics
            self.metrics.add_cpu_sample(cpu_percent)
            self.metrics.add_memory_sample(memory_mb)
            self.metrics.update_generation(self.current_entities, generation_rate)
            self.metrics.disk_read_total_mb = disk_read_mb
            self.metrics.disk_write_total_mb = disk_write_mb
            
            # Store snapshot
            self.snapshots.append(snapshot)
            self.last_snapshot_time = time.time()
            
            return snapshot
            
        except Exception as e:
            logger.warning(f"Failed to take performance snapshot: {e}")
            return None
    
    def get_current_stats(self) -> Dict[str, Any]:
        """
        Get current performance statistics.
        
        Returns:
            Dictionary with current stats
        """
        if not self.enabled:
            return {}
        
        snapshot = self.take_snapshot()
        if snapshot:
            return {
                "cpu_percent": snapshot.cpu_percent,
                "memory_mb": snapshot.memory_mb,
                "entities": snapshot.entities_generated,
                "rate": snapshot.generation_rate,
            }
        return {}
    
    def get_warnings(self) -> List[str]:
        """
        Get performance warnings.
        
        Returns:
            List of warning messages
        """
        return self.metrics.warnings.copy()
    
    def finalize(self) -> None:
        """Finalize monitoring."""
        if self.enabled:
            self.metrics.finalize()
            logger.info(
                f"Performance monitoring complete: "
                f"{self.metrics.total_entities} entities in "
                f"{self.metrics.duration_seconds:.2f}s "
                f"(avg rate: {self.metrics.generation_rate_avg:.1f}/s)"
            )
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get performance summary.
        
        Returns:
            Dictionary with summary statistics
        """
        return self.metrics.to_dict()
    
    def export_report(self, output_file: Path) -> None:
        """
        Export performance report to JSON file.
        
        Args:
            output_file: Path to output file
        """
        if not self.enabled:
            logger.warning("Performance monitoring disabled, cannot export report")
            return
        
        report = {
            "metadata": {
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "end_time": datetime.fromtimestamp(self.metrics.end_time or time.time()).isoformat(),
                "duration_seconds": self.metrics.duration_seconds,
                "sample_interval": self.sample_interval,
                "total_snapshots": len(self.snapshots),
            },
            "summary": self.get_summary(),
            "snapshots": [s.to_dict() for s in self.snapshots],
        }
        
        # Ensure directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Performance report exported to: {output_file}")
    
    def print_summary(self, console=None) -> None:
        """
        Print performance summary.
        
        Args:
            console: Rich console instance (optional)
        """
        from rich.console import Console
        from rich.table import Table
        
        if console is None:
            console = Console()
        
        if not self.enabled:
            console.print("[yellow]Performance monitoring was disabled[/yellow]")
            return
        
        summary = self.get_summary()
        
        table = Table(title="Performance Summary", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="yellow")
        
        # Duration
        table.add_row("Duration", f"{summary['duration_seconds']:.2f}s")
        
        # CPU
        table.add_section()
        table.add_row("[bold]CPU[/bold]", "")
        table.add_row("  Minimum", f"{summary['cpu']['min_percent']:.1f}%")
        table.add_row("  Average", f"{summary['cpu']['avg_percent']:.1f}%")
        table.add_row("  Maximum", f"{summary['cpu']['max_percent']:.1f}%")
        
        # Memory
        table.add_section()
        table.add_row("[bold]Memory[/bold]", "")
        table.add_row("  Minimum", f"{summary['memory']['min_mb']:.1f} MB")
        table.add_row("  Average", f"{summary['memory']['avg_mb']:.1f} MB")
        table.add_row("  Peak", f"{summary['memory']['peak_mb']:.1f} MB")
        
        # Disk I/O
        table.add_section()
        table.add_row("[bold]Disk I/O[/bold]", "")
        table.add_row("  Read", f"{summary['disk_io']['read_total_mb']:.1f} MB")
        table.add_row("  Write", f"{summary['disk_io']['write_total_mb']:.1f} MB")
        
        # Generation
        table.add_section()
        table.add_row("[bold]Generation[/bold]", "")
        table.add_row("  Total Entities", f"{summary['generation']['total_entities']:,}")
        table.add_row("  Avg Rate", f"{summary['generation']['avg_rate']:.1f}/s")
        table.add_row("  Peak Rate", f"{summary['generation']['peak_rate']:.1f}/s")
        
        console.print(table)
        
        # Warnings
        if self.metrics.warnings:
            console.print("\n[yellow bold]Performance Warnings:[/yellow bold]")
            for warning in self.metrics.warnings[:10]:  # Limit to 10
                console.print(f"  [yellow]⚠[/yellow] {warning}")
            if len(self.metrics.warnings) > 10:
                console.print(f"  [dim]... and {len(self.metrics.warnings) - 10} more[/dim]")


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create the global performance monitor."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def initialize_monitor(enabled: bool = True, sample_interval: float = 1.0) -> PerformanceMonitor:
    """
    Initialize global performance monitor.
    
    Args:
        enabled: Whether monitoring is enabled
        sample_interval: Seconds between samples
    
    Returns:
        Initialized PerformanceMonitor instance
    """
    global _performance_monitor
    _performance_monitor = PerformanceMonitor(enabled=enabled, sample_interval=sample_interval)
    return _performance_monitor
