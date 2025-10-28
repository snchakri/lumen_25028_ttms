"""
Progress Tracking System for PyGMO Solver Family

This module implements real-time progress tracking for optimization execution,
providing visual feedback and performance monitoring.

Theoretical Foundation: Section 10.2 - Solution Pipeline
"""

import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class ProgressMetrics:
    """Metrics for progress tracking"""
    current_generation: int = 0
    total_generations: int = 0
    current_hypervolume: float = 0.0
    best_hypervolume: float = 0.0
    stagnation_count: int = 0
    evaluations: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    last_update_time: datetime = field(default_factory=datetime.now)
    
    def elapsed_time(self) -> timedelta:
        """Calculate elapsed time"""
        return datetime.now() - self.start_time
    
    def estimated_remaining_time(self) -> Optional[timedelta]:
        """Estimate remaining time based on current progress"""
        if self.current_generation == 0:
            return None
        
        elapsed = self.elapsed_time()
        progress_ratio = self.current_generation / self.total_generations
        
        if progress_ratio > 0:
            total_estimated = elapsed / progress_ratio
            remaining = total_estimated - elapsed
            return remaining
        
        return None
    
    def progress_percentage(self) -> float:
        """Calculate progress percentage"""
        if self.total_generations == 0:
            return 0.0
        return (self.current_generation / self.total_generations) * 100.0
    
    def generations_per_second(self) -> float:
        """Calculate generation rate"""
        elapsed = self.elapsed_time().total_seconds()
        if elapsed > 0:
            return self.current_generation / elapsed
        return 0.0


class ProgressTracker:
    """
    Real-time progress tracking for PyGMO optimization.
    
    Features:
    - Generation progress tracking
    - Hypervolume evolution monitoring
    - Time estimation
    - Performance metrics
    - Convergence detection
    """
    
    def __init__(
        self,
        total_generations: int,
        update_frequency: int = 10,
        logger: Optional[Any] = None
    ):
        """
        Initialize progress tracker.
        
        Args:
            total_generations: Total number of generations
            update_frequency: Update console every N generations
            logger: Optional StructuredLogger instance
        """
        self.metrics = ProgressMetrics(total_generations=total_generations)
        self.update_frequency = update_frequency
        self.logger = logger
        
        # History tracking
        self.hypervolume_history: List[float] = []
        self.generation_times: List[float] = []
        
        # Island tracking
        self.island_metrics: Dict[int, Dict[str, Any]] = {}
        
        # Convergence tracking
        self.convergence_detected = False
        self.convergence_generation: Optional[int] = None
    
    def start(self):
        """Start progress tracking"""
        self.metrics.start_time = datetime.now()
        self.metrics.last_update_time = datetime.now()
        
        if self.logger:
            self.logger.info(
                f"Starting optimization: {self.metrics.total_generations} generations",
                total_generations=self.metrics.total_generations
            )
        else:
            print(f"\n{'='*80}")
            print(f"PyGMO Optimization Started")
            print(f"Total Generations: {self.metrics.total_generations}")
            print(f"{'='*80}\n")
    
    def update(
        self,
        generation: int,
        hypervolume: float,
        best_fitness: Optional[List[float]] = None,
        **kwargs
    ):
        """
        Update progress with current generation metrics.
        
        Args:
            generation: Current generation number
            hypervolume: Current hypervolume indicator
            best_fitness: Best fitness values
            **kwargs: Additional metrics
        """
        self.metrics.current_generation = generation
        self.metrics.current_hypervolume = hypervolume
        self.metrics.evaluations = kwargs.get('evaluations', 0)
        
        # Update best hypervolume
        if hypervolume > self.metrics.best_hypervolume:
            self.metrics.best_hypervolume = hypervolume
            self.metrics.stagnation_count = 0
        else:
            self.metrics.stagnation_count += 1
        
        # Track history
        self.hypervolume_history.append(hypervolume)
        
        # Calculate generation time
        current_time = datetime.now()
        gen_time = (current_time - self.metrics.last_update_time).total_seconds()
        self.generation_times.append(gen_time)
        self.metrics.last_update_time = current_time
        
        # Display progress
        if generation % self.update_frequency == 0 or generation == self.metrics.total_generations:
            self._display_progress(best_fitness, **kwargs)
        
        # Log progress
        if self.logger:
            self.logger.log_optimization_progress(
                generation=generation,
                hypervolume=hypervolume,
                best_fitness=best_fitness or [],
                **kwargs
            )
    
    def update_island(self, island_id: int, metrics: Dict[str, Any]):
        """
        Update metrics for specific island.
        
        Args:
            island_id: Island identifier
            metrics: Island-specific metrics
        """
        self.island_metrics[island_id] = {
            **metrics,
            "timestamp": datetime.now().isoformat()
        }
    
    def detect_convergence(self, tolerance: float = 1e-4, window: int = 50) -> bool:
        """
        Detect convergence based on hypervolume stagnation.
        
        Args:
            tolerance: Hypervolume change tolerance
            window: Number of generations to check
        
        Returns:
            True if converged
        """
        if len(self.hypervolume_history) < window:
            return False
        
        recent_hv = self.hypervolume_history[-window:]
        hv_change = max(recent_hv) - min(recent_hv)
        
        if hv_change < tolerance:
            if not self.convergence_detected:
                self.convergence_detected = True
                self.convergence_generation = self.metrics.current_generation
                
                if self.logger:
                    self.logger.info(
                        f"Convergence detected at generation {self.convergence_generation}",
                        generation=self.convergence_generation,
                        hypervolume_change=hv_change
                    )
                else:
                    print(f"\n{'='*80}")
                    print(f"Convergence Detected at Generation {self.convergence_generation}")
                    print(f"Hypervolume Change: {hv_change:.6e}")
                    print(f"{'='*80}\n")
            
            return True
        
        return False
    
    def _display_progress(self, best_fitness: Optional[List[float]], **kwargs):
        """Display progress to console"""
        progress_pct = self.metrics.progress_percentage()
        elapsed = self.metrics.elapsed_time()
        remaining = self.metrics.estimated_remaining_time()
        gen_per_sec = self.metrics.generations_per_second()
        
        # Progress bar
        bar_length = 40
        filled_length = int(bar_length * progress_pct / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Format time
        elapsed_str = str(elapsed).split('.')[0]  # Remove microseconds
        remaining_str = str(remaining).split('.')[0] if remaining else "N/A"
        
        # Display
        print(f"\r[{bar}] {progress_pct:5.1f}% | "
              f"Gen: {self.metrics.current_generation}/{self.metrics.total_generations} | "
              f"HV: {self.metrics.current_hypervolume:.6f} | "
              f"Best HV: {self.metrics.best_hypervolume:.6f} | "
              f"Stag: {self.metrics.stagnation_count} | "
              f"Time: {elapsed_str} | "
              f"ETA: {remaining_str} | "
              f"Rate: {gen_per_sec:.2f} gen/s", end='', flush=True)
        
        # Newline at completion or every N updates
        if self.metrics.current_generation == self.metrics.total_generations:
            print()  # Newline at end
    
    def finish(self, status: str = "COMPLETED"):
        """
        Finish progress tracking and display summary.
        
        Args:
            status: Completion status
        """
        elapsed = self.metrics.elapsed_time()
        
        summary = {
            "status": status,
            "total_generations": self.metrics.current_generation,
            "final_hypervolume": self.metrics.current_hypervolume,
            "best_hypervolume": self.metrics.best_hypervolume,
            "total_evaluations": self.metrics.evaluations,
            "elapsed_time_seconds": elapsed.total_seconds(),
            "average_gen_per_second": self.metrics.generations_per_second(),
            "convergence_detected": self.convergence_detected,
            "convergence_generation": self.convergence_generation,
        }
        
        if self.logger:
            self.logger.info(
                f"Optimization {status}",
                **summary
            )
        else:
            print(f"\n{'='*80}")
            print(f"Optimization {status}")
            print(f"{'='*80}")
            print(f"Total Generations: {summary['total_generations']}")
            print(f"Final Hypervolume: {summary['final_hypervolume']:.6f}")
            print(f"Best Hypervolume: {summary['best_hypervolume']:.6f}")
            print(f"Total Evaluations: {summary['total_evaluations']}")
            print(f"Elapsed Time: {elapsed}")
            print(f"Average Rate: {summary['average_gen_per_second']:.2f} gen/s")
            if self.convergence_detected:
                print(f"Converged at Generation: {self.convergence_generation}")
            print(f"{'='*80}\n")
        
        return summary
    
    def get_summary(self) -> Dict[str, Any]:
        """Get progress summary"""
        return {
            "current_generation": self.metrics.current_generation,
            "total_generations": self.metrics.total_generations,
            "progress_percentage": self.metrics.progress_percentage(),
            "current_hypervolume": self.metrics.current_hypervolume,
            "best_hypervolume": self.metrics.best_hypervolume,
            "stagnation_count": self.metrics.stagnation_count,
            "elapsed_time_seconds": self.metrics.elapsed_time().total_seconds(),
            "estimated_remaining_seconds": self.metrics.estimated_remaining_time().total_seconds() if self.metrics.estimated_remaining_time() else None,
            "generations_per_second": self.metrics.generations_per_second(),
            "convergence_detected": self.convergence_detected,
            "island_count": len(self.island_metrics),
        }


