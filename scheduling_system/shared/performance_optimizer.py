"""
Performance Optimization System
Comprehensive performance analysis and optimization for the 7-stage scheduling engine
with memory constraints compliance and Railway deployment optimization.
"""

import psutil
import time
import gc
import tracemalloc
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
import json
import numpy as np
import pandas as pd

logger = structlog.get_logger(__name__)

class PerformanceConstraint(str, Enum):
    """Performance constraints for Railway deployment."""
    MAX_MEMORY_MB = 512  # 512MB memory limit
    MAX_EXECUTION_TIME_S = 600  # 10 minutes execution time limit
    MAX_STAGE_TIME_S = 120  # 2 minutes per stage limit
    MAX_FILE_SIZE_MB = 100  # 100MB file size limit

class PerformanceLevel(str, Enum):
    """Performance levels for optimization."""
    OPTIMAL = "optimal"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"

@dataclass
class PerformanceMetrics:
    """Performance metrics for a component or stage."""
    execution_time: float
    memory_usage_mb: float
    peak_memory_mb: float
    cpu_usage_percent: float
    file_size_mb: float
    throughput_items_per_sec: float
    error_rate: float
    cache_hit_rate: float = 0.0
    optimization_score: float = 0.0

@dataclass
class PerformanceAnalysis:
    """Comprehensive performance analysis results."""
    stage: int
    component: str
    metrics: PerformanceMetrics
    constraints_violated: List[str]
    optimization_recommendations: List[str]
    performance_level: PerformanceLevel
    railway_compliant: bool
    bottlenecks: List[str]
    memory_efficiency: float
    time_efficiency: float

class PerformanceOptimizer:
    """
    Comprehensive performance optimizer for the 7-stage scheduling engine.
    Analyzes performance bottlenecks and provides optimization recommendations.
    """
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("performance_analysis")
        self.output_dir.mkdir(exist_ok=True)
        
        # Performance tracking
        self.performance_history: List[PerformanceAnalysis] = []
        self.memory_snapshots: List[Dict[str, Any]] = []
        self.bottleneck_analysis: Dict[str, List[str]] = {}
        
        # Initialize memory tracking
        tracemalloc.start()
        
        self.logger = logger.bind(component="PerformanceOptimizer")
        self.logger.info("Performance optimizer initialized")
    
    def analyze_stage_performance(self, stage: int, component: str, 
                                execution_data: Dict[str, Any]) -> PerformanceAnalysis:
        """
        Analyze performance of a specific stage component.
        
        Args:
            stage: Stage number (1-7)
            component: Component name
            execution_data: Execution data including timing and memory info
            
        Returns:
            Comprehensive performance analysis
        """
        self.logger.info(f"Analyzing performance for stage {stage}, component {component}")
        
        # Extract metrics from execution data
        metrics = self._extract_performance_metrics(execution_data)
        
        # Analyze constraints
        constraints_violated = self._check_constraints(metrics)
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(metrics, execution_data)
        
        # Generate optimization recommendations
        recommendations = self._generate_recommendations(metrics, constraints_violated, bottlenecks)
        
        # Calculate performance level
        performance_level = self._calculate_performance_level(metrics, constraints_violated)
        
        # Check Railway compliance
        railway_compliant = len(constraints_violated) == 0
        
        # Calculate efficiency scores
        memory_efficiency = self._calculate_memory_efficiency(metrics)
        time_efficiency = self._calculate_time_efficiency(metrics)
        
        # Create analysis
        analysis = PerformanceAnalysis(
            stage=stage,
            component=component,
            metrics=metrics,
            constraints_violated=constraints_violated,
            optimization_recommendations=recommendations,
            performance_level=performance_level,
            railway_compliant=railway_compliant,
            bottlenecks=bottlenecks,
            memory_efficiency=memory_efficiency,
            time_efficiency=time_efficiency
        )
        
        # Store analysis
        self.performance_history.append(analysis)
        
        # Update bottleneck tracking
        if bottlenecks:
            self.bottleneck_analysis[f"stage_{stage}_{component}"] = bottlenecks
        
        self.logger.info(f"Performance analysis completed for stage {stage}, component {component}",
                        performance_level=performance_level.value,
                        railway_compliant=railway_compliant,
                        bottlenecks_count=len(bottlenecks))
        
        return analysis
    
    def _extract_performance_metrics(self, execution_data: Dict[str, Any]) -> PerformanceMetrics:
        """Extract performance metrics from execution data."""
        return PerformanceMetrics(
            execution_time=execution_data.get("execution_time", 0.0),
            memory_usage_mb=execution_data.get("memory_usage_mb", 0.0),
            peak_memory_mb=execution_data.get("peak_memory_mb", 0.0),
            cpu_usage_percent=execution_data.get("cpu_usage_percent", 0.0),
            file_size_mb=execution_data.get("file_size_mb", 0.0),
            throughput_items_per_sec=execution_data.get("throughput_items_per_sec", 0.0),
            error_rate=execution_data.get("error_rate", 0.0)
        )
    
    def _check_constraints(self, metrics: PerformanceMetrics) -> List[str]:
        """Check performance constraints and return violations."""
        violations = []
        
        # Memory constraints
        if metrics.peak_memory_mb > PerformanceConstraint.MAX_MEMORY_MB.value:
            violations.append(f"Peak memory {metrics.peak_memory_mb:.2f}MB exceeds {PerformanceConstraint.MAX_MEMORY_MB.value}MB limit")
        
        if metrics.memory_usage_mb > PerformanceConstraint.MAX_MEMORY_MB.value:
            violations.append(f"Memory usage {metrics.memory_usage_mb:.2f}MB exceeds {PerformanceConstraint.MAX_MEMORY_MB.value}MB limit")
        
        # Time constraints
        if metrics.execution_time > PerformanceConstraint.MAX_EXECUTION_TIME_S.value:
            violations.append(f"Execution time {metrics.execution_time:.2f}s exceeds {PerformanceConstraint.MAX_EXECUTION_TIME_S.value}s limit")
        
        if metrics.execution_time > PerformanceConstraint.MAX_STAGE_TIME_S.value:
            violations.append(f"Stage time {metrics.execution_time:.2f}s exceeds {PerformanceConstraint.MAX_STAGE_TIME_S.value}s limit")
        
        # File size constraints
        if metrics.file_size_mb > PerformanceConstraint.MAX_FILE_SIZE_MB.value:
            violations.append(f"File size {metrics.file_size_mb:.2f}MB exceeds {PerformanceConstraint.MAX_FILE_SIZE_MB.value}MB limit")
        
        return violations
    
    def _identify_bottlenecks(self, metrics: PerformanceMetrics, 
                            execution_data: Dict[str, Any]) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        # Memory bottlenecks
        if metrics.memory_usage_mb > PerformanceConstraint.MAX_MEMORY_MB.value * 0.8:
            bottlenecks.append("High memory usage - consider memory optimization")
        
        if metrics.peak_memory_mb > metrics.memory_usage_mb * 2:
            bottlenecks.append("Memory spikes detected - check for memory leaks")
        
        # CPU bottlenecks
        if metrics.cpu_usage_percent > 80:
            bottlenecks.append("High CPU usage - consider algorithm optimization")
        
        # I/O bottlenecks
        if execution_data.get("io_operations", 0) > 1000:
            bottlenecks.append("High I/O operations - consider caching or batch processing")
        
        # Time bottlenecks
        if metrics.execution_time > PerformanceConstraint.MAX_STAGE_TIME_S.value * 0.5:
            bottlenecks.append("Slow execution - consider algorithm optimization")
        
        # Throughput bottlenecks
        if metrics.throughput_items_per_sec < 10:
            bottlenecks.append("Low throughput - consider batch processing optimization")
        
        return bottlenecks
    
    def _generate_recommendations(self, metrics: PerformanceMetrics,
                                constraints_violated: List[str],
                                bottlenecks: List[str]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Memory optimization recommendations
        if any("memory" in violation.lower() for violation in constraints_violated):
            recommendations.extend([
                "Implement memory-efficient data structures",
                "Use generators instead of lists for large datasets",
                "Clear unused variables with del statement",
                "Consider using memory-mapped files for large datasets",
                "Implement data chunking for processing large files"
            ])
        
        # Time optimization recommendations
        if any("time" in violation.lower() or "execution" in violation.lower() for violation in constraints_violated):
            recommendations.extend([
                "Profile code to identify slow functions",
                "Use vectorized operations with NumPy/Pandas",
                "Implement caching for repeated computations",
                "Consider parallel processing for independent operations",
                "Optimize database queries and reduce I/O operations"
            ])
        
        # Algorithm-specific recommendations
        if metrics.throughput_items_per_sec < 50:
            recommendations.extend([
                "Consider using more efficient algorithms",
                "Implement early termination conditions",
                "Use approximate algorithms for large datasets",
                "Cache intermediate results"
            ])
        
        # Railway-specific recommendations
        if len(constraints_violated) > 0:
            recommendations.extend([
                "Optimize for Railway deployment constraints",
                "Implement graceful degradation for resource limits",
                "Use streaming processing for large datasets",
                "Consider external storage for large intermediate files"
            ])
        
        return recommendations
    
    def _calculate_performance_level(self, metrics: PerformanceMetrics,
                                   constraints_violated: List[str]) -> PerformanceLevel:
        """Calculate overall performance level."""
        if len(constraints_violated) > 2:
            return PerformanceLevel.CRITICAL
        elif len(constraints_violated) > 0:
            return PerformanceLevel.POOR
        elif metrics.execution_time > PerformanceConstraint.MAX_STAGE_TIME_S.value * 0.5:
            return PerformanceLevel.ACCEPTABLE
        elif metrics.memory_usage_mb > PerformanceConstraint.MAX_MEMORY_MB.value * 0.5:
            return PerformanceLevel.GOOD
        else:
            return PerformanceLevel.OPTIMAL
    
    def _calculate_memory_efficiency(self, metrics: PerformanceMetrics) -> float:
        """Calculate memory efficiency score (0-1)."""
        if metrics.peak_memory_mb == 0:
            return 1.0
        
        efficiency = 1.0 - (metrics.peak_memory_mb / PerformanceConstraint.MAX_MEMORY_MB.value)
        return max(0.0, min(1.0, efficiency))
    
    def _calculate_time_efficiency(self, metrics: PerformanceMetrics) -> float:
        """Calculate time efficiency score (0-1)."""
        if metrics.execution_time == 0:
            return 1.0
        
        efficiency = 1.0 - (metrics.execution_time / PerformanceConstraint.MAX_STAGE_TIME_S.value)
        return max(0.0, min(1.0, efficiency))
    
    def analyze_system_performance(self) -> Dict[str, Any]:
        """Analyze overall system performance."""
        self.logger.info("Analyzing overall system performance")
        
        if not self.performance_history:
            return {"error": "No performance data available"}
        
        # Aggregate metrics
        total_execution_time = sum(analysis.metrics.execution_time for analysis in self.performance_history)
        total_memory_usage = sum(analysis.metrics.memory_usage_mb for analysis in self.performance_history)
        max_peak_memory = max(analysis.metrics.peak_memory_mb for analysis in self.performance_history)
        
        # Performance levels distribution
        performance_levels = {}
        for analysis in self.performance_history:
            level = analysis.performance_level.value
            performance_levels[level] = performance_levels.get(level, 0) + 1
        
        # Railway compliance
        railway_compliant_stages = sum(1 for analysis in self.performance_history if analysis.railway_compliant)
        railway_compliance_rate = railway_compliant_stages / len(self.performance_history)
        
        # Bottleneck analysis
        all_bottlenecks = []
        for bottlenecks in self.bottleneck_analysis.values():
            all_bottlenecks.extend(bottlenecks)
        
        bottleneck_frequency = {}
        for bottleneck in all_bottlenecks:
            bottleneck_frequency[bottleneck] = bottleneck_frequency.get(bottleneck, 0) + 1
        
        # System analysis
        system_analysis = {
            "total_stages_analyzed": len(self.performance_history),
            "total_execution_time": total_execution_time,
            "total_memory_usage": total_memory_usage,
            "max_peak_memory": max_peak_memory,
            "performance_levels_distribution": performance_levels,
            "railway_compliance_rate": railway_compliance_rate,
            "railway_compliant": railway_compliance_rate >= 0.8,
            "bottleneck_frequency": bottleneck_frequency,
            "critical_issues": [
                analysis for analysis in self.performance_history 
                if analysis.performance_level == PerformanceLevel.CRITICAL
            ],
            "optimization_priority": self._calculate_optimization_priority(),
            "system_efficiency_score": self._calculate_system_efficiency_score()
        }
        
        # Save system analysis
        analysis_path = self.output_dir / "system_performance_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(system_analysis, f, indent=2, default=str)
        
        self.logger.info("System performance analysis completed",
                        railway_compliant=system_analysis["railway_compliant"],
                        system_efficiency=system_analysis["system_efficiency_score"])
        
        return system_analysis
    
    def _calculate_optimization_priority(self) -> List[str]:
        """Calculate optimization priority based on performance analysis."""
        priorities = []
        
        # Count critical issues
        critical_count = sum(1 for analysis in self.performance_history 
                           if analysis.performance_level == PerformanceLevel.CRITICAL)
        
        if critical_count > 0:
            priorities.append(f"CRITICAL: {critical_count} stages have critical performance issues")
        
        # Memory optimization priority
        high_memory_count = sum(1 for analysis in self.performance_history 
                              if analysis.metrics.memory_usage_mb > PerformanceConstraint.MAX_MEMORY_MB.value * 0.7)
        
        if high_memory_count > 0:
            priorities.append(f"HIGH: {high_memory_count} stages have high memory usage")
        
        # Time optimization priority
        slow_stages = sum(1 for analysis in self.performance_history 
                         if analysis.metrics.execution_time > PerformanceConstraint.MAX_STAGE_TIME_S.value * 0.5)
        
        if slow_stages > 0:
            priorities.append(f"MEDIUM: {slow_stages} stages are slow")
        
        return priorities
    
    def _calculate_system_efficiency_score(self) -> float:
        """Calculate overall system efficiency score (0-1)."""
        if not self.performance_history:
            return 0.0
        
        memory_scores = [analysis.memory_efficiency for analysis in self.performance_history]
        time_scores = [analysis.time_efficiency for analysis in self.performance_history]
        
        avg_memory_efficiency = np.mean(memory_scores)
        avg_time_efficiency = np.mean(time_scores)
        
        # Weighted average (memory and time equally important)
        overall_efficiency = (avg_memory_efficiency + avg_time_efficiency) / 2
        
        return overall_efficiency
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        self.logger.info("Generating optimization report")
        
        # System analysis
        system_analysis = self.analyze_system_performance()
        
        # Stage-by-stage analysis
        stage_analyses = {}
        for analysis in self.performance_history:
            stage_key = f"stage_{analysis.stage}_{analysis.component}"
            stage_analyses[stage_key] = asdict(analysis)
        
        # Optimization recommendations
        all_recommendations = []
        for analysis in self.performance_history:
            all_recommendations.extend(analysis.optimization_recommendations)
        
        # Remove duplicates and prioritize
        unique_recommendations = list(set(all_recommendations))
        
        # Create comprehensive report
        optimization_report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "system_analysis": system_analysis,
            "stage_analyses": stage_analyses,
            "optimization_recommendations": {
                "unique_recommendations": unique_recommendations,
                "total_recommendations": len(all_recommendations),
                "priority_recommendations": self._get_priority_recommendations()
            },
            "railway_deployment": {
                "compliant": system_analysis["railway_compliant"],
                "compliance_rate": system_analysis["railway_compliance_rate"],
                "constraints_summary": self._get_constraints_summary()
            },
            "performance_summary": {
                "overall_efficiency": system_analysis["system_efficiency_score"],
                "critical_issues_count": len(system_analysis["critical_issues"]),
                "optimization_priority": system_analysis["optimization_priority"]
            }
        }
        
        # Save optimization report
        report_path = self.output_dir / "optimization_report.json"
        with open(report_path, 'w') as f:
            json.dump(optimization_report, f, indent=2, default=str)
        
        self.logger.info("Optimization report generated", 
                        report_path=str(report_path),
                        railway_compliant=system_analysis["railway_compliant"])
        
        return optimization_report
    
    def _get_priority_recommendations(self) -> List[str]:
        """Get priority optimization recommendations."""
        # Count recommendation frequency
        recommendation_counts = {}
        for analysis in self.performance_history:
            for rec in analysis.optimization_recommendations:
                recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
        
        # Sort by frequency
        sorted_recommendations = sorted(recommendation_counts.items(), 
                                      key=lambda x: x[1], reverse=True)
        
        return [rec for rec, count in sorted_recommendations[:10]]  # Top 10
    
    def _get_constraints_summary(self) -> Dict[str, Any]:
        """Get summary of constraint violations."""
        constraints_summary = {
            "memory_violations": 0,
            "time_violations": 0,
            "file_size_violations": 0,
            "total_violations": 0
        }
        
        for analysis in self.performance_history:
            for violation in analysis.constraints_violated:
                constraints_summary["total_violations"] += 1
                
                if "memory" in violation.lower():
                    constraints_summary["memory_violations"] += 1
                elif "time" in violation.lower() or "execution" in violation.lower():
                    constraints_summary["time_violations"] += 1
                elif "file" in violation.lower():
                    constraints_summary["file_size_violations"] += 1
        
        return constraints_summary

# Factory function
def create_performance_optimizer(output_dir: Path = None) -> PerformanceOptimizer:
    """
    Create a performance optimizer.
    
    Args:
        output_dir: Directory for performance analysis outputs
        
    Returns:
        Configured PerformanceOptimizer
    """
    return PerformanceOptimizer(output_dir)

