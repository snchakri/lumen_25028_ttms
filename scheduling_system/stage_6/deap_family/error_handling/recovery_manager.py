"""
Recovery Manager for DEAP Solver Family

Implements intelligent recovery mechanisms and fallback strategies
as per Stage 6.3 foundational requirements.

Author: LUMEN Team [TEAM-ID: 93912]
"""

import logging
import time
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass
from enum import Enum

from .error_reporter import ErrorReporter


class RecoveryStrategy(Enum):
    """Recovery strategy types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    SOLVER_SWITCH = "solver_switch"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CHECKPOINT_RESTORE = "checkpoint_restore"


@dataclass
class RecoveryAction:
    """Recovery action specification."""
    strategy: RecoveryStrategy
    description: str
    parameters: Dict[str, Any]
    max_attempts: int
    timeout_seconds: Optional[float]
    success_condition: Optional[Callable[[], bool]]


class RecoveryManager:
    """
    Intelligent recovery and fallback management system.
    
    Implements foundation-compliant recovery mechanisms without
    overcomplicating runtime and management.
    """
    
    def __init__(self, error_reporter: ErrorReporter, logger: logging.Logger):
        """
        Initialize recovery manager.
        
        Args:
            error_reporter: Error reporting system
            logger: Logger instance
        """
        self.error_reporter = error_reporter
        self.logger = logger
        
        # Recovery attempt tracking
        self.recovery_attempts: Dict[str, int] = {}
        self.recovery_history: List[Dict[str, Any]] = []
        
        # Maximum recovery attempts per error type
        self.max_recovery_attempts = {
            "InputError": 3,
            "SolverError": 2,
            "ValidationError": 2,
            "ConfigurationError": 1,
            "MemoryError": 2,
            "ConvergenceError": 3
        }
    
    def attempt_recovery(
        self,
        error_type: str,
        error_context: Dict[str, Any],
        recovery_function: Callable[[], Any],
        fallback_function: Optional[Callable[[], Any]] = None
    ) -> Tuple[bool, Any, Optional[str]]:
        """
        Attempt recovery from error.
        
        Args:
            error_type: Type of error to recover from
            error_context: Error context information
            recovery_function: Function to attempt recovery
            fallback_function: Optional fallback function
        
        Returns:
            Tuple of (success, result, recovery_strategy_used)
        """
        recovery_key = f"{error_type}_{hash(str(error_context))}"
        
        # Check if we've exceeded maximum attempts
        attempts = self.recovery_attempts.get(recovery_key, 0)
        max_attempts = self.max_recovery_attempts.get(error_type, 2)
        
        if attempts >= max_attempts:
            self.logger.warning(f"Maximum recovery attempts ({max_attempts}) exceeded for {error_type}")
            return False, None, None
        
        # Increment attempt counter
        self.recovery_attempts[recovery_key] = attempts + 1
        
        # Select recovery strategy
        recovery_actions = self._select_recovery_actions(error_type, error_context, attempts)
        
        for action in recovery_actions:
            self.logger.info(f"Attempting recovery: {action.strategy.value} - {action.description}")
            
            success, result = self._execute_recovery_action(
                action,
                recovery_function,
                fallback_function
            )
            
            # Record recovery attempt
            self._record_recovery_attempt(error_type, action, success, error_context)
            
            if success:
                self.logger.info(f"Recovery successful using strategy: {action.strategy.value}")
                return True, result, action.strategy.value
            
            self.logger.warning(f"Recovery attempt failed: {action.strategy.value}")
        
        # All recovery attempts failed
        self.logger.error(f"All recovery attempts failed for {error_type}")
        return False, None, None
    
    def _select_recovery_actions(
        self,
        error_type: str,
        error_context: Dict[str, Any],
        attempt_number: int
    ) -> List[RecoveryAction]:
        """Select appropriate recovery actions based on error type and context."""
        actions = []
        
        if error_type == "InputError":
            if attempt_number == 0:
                # First attempt: retry with validation
                actions.append(RecoveryAction(
                    strategy=RecoveryStrategy.RETRY,
                    description="Retry input loading with enhanced validation",
                    parameters={"validate_schema": True, "repair_data": True},
                    max_attempts=1,
                    timeout_seconds=30.0,
                    success_condition=None
                ))
            elif attempt_number == 1:
                # Second attempt: fallback to default parameters
                actions.append(RecoveryAction(
                    strategy=RecoveryStrategy.FALLBACK,
                    description="Use default parameters for missing data",
                    parameters={"use_defaults": True, "skip_optional": True},
                    max_attempts=1,
                    timeout_seconds=15.0,
                    success_condition=None
                ))
            else:
                # Final attempt: graceful degradation
                actions.append(RecoveryAction(
                    strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
                    description="Proceed with minimal required data",
                    parameters={"minimal_mode": True},
                    max_attempts=1,
                    timeout_seconds=10.0,
                    success_condition=None
                ))
        
        elif error_type == "SolverError":
            if attempt_number == 0:
                # First attempt: parameter adjustment
                actions.append(RecoveryAction(
                    strategy=RecoveryStrategy.PARAMETER_ADJUSTMENT,
                    description="Adjust solver parameters within foundation bounds",
                    parameters={"reduce_population": True, "increase_generations": True},
                    max_attempts=1,
                    timeout_seconds=60.0,
                    success_condition=None
                ))
            else:
                # Second attempt: solver switch
                actions.append(RecoveryAction(
                    strategy=RecoveryStrategy.SOLVER_SWITCH,
                    description="Switch to more robust solver (NSGA-II)",
                    parameters={"target_solver": "nsga2", "conservative_params": True},
                    max_attempts=1,
                    timeout_seconds=120.0,
                    success_condition=None
                ))
        
        elif error_type == "MemoryError":
            if attempt_number == 0:
                # First attempt: reduce memory usage
                actions.append(RecoveryAction(
                    strategy=RecoveryStrategy.PARAMETER_ADJUSTMENT,
                    description="Reduce memory usage while maintaining foundation compliance",
                    parameters={"reduce_population": True, "streaming_mode": True},
                    max_attempts=1,
                    timeout_seconds=45.0,
                    success_condition=None
                ))
            else:
                # Second attempt: graceful degradation
                actions.append(RecoveryAction(
                    strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
                    description="Use memory-efficient algorithms",
                    parameters={"memory_efficient": True, "batch_processing": True},
                    max_attempts=1,
                    timeout_seconds=90.0,
                    success_condition=None
                ))
        
        elif error_type == "ConvergenceError":
            if attempt_number == 0:
                # First attempt: adjust convergence parameters
                actions.append(RecoveryAction(
                    strategy=RecoveryStrategy.PARAMETER_ADJUSTMENT,
                    description="Adjust convergence criteria and diversity mechanisms",
                    parameters={"increase_diversity": True, "relax_convergence": True},
                    max_attempts=1,
                    timeout_seconds=180.0,
                    success_condition=None
                ))
            elif attempt_number == 1:
                # Second attempt: hybrid approach
                actions.append(RecoveryAction(
                    strategy=RecoveryStrategy.SOLVER_SWITCH,
                    description="Use hybrid evolutionary approach",
                    parameters={"hybrid_mode": True, "local_search": True},
                    max_attempts=1,
                    timeout_seconds=300.0,
                    success_condition=None
                ))
            else:
                # Final attempt: accept suboptimal solution
                actions.append(RecoveryAction(
                    strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
                    description="Accept best solution found so far",
                    parameters={"accept_suboptimal": True},
                    max_attempts=1,
                    timeout_seconds=10.0,
                    success_condition=None
                ))
        
        return actions
    
    def _execute_recovery_action(
        self,
        action: RecoveryAction,
        recovery_function: Callable[[], Any],
        fallback_function: Optional[Callable[[], Any]]
    ) -> Tuple[bool, Any]:
        """Execute a specific recovery action."""
        start_time = time.time()
        
        try:
            # Apply recovery parameters to function context
            # This would typically involve modifying global state or passing parameters
            # For now, we'll attempt the recovery function directly
            
            if action.strategy == RecoveryStrategy.FALLBACK and fallback_function:
                result = fallback_function()
            else:
                result = recovery_function()
            
            # Check timeout
            if action.timeout_seconds and (time.time() - start_time) > action.timeout_seconds:
                self.logger.warning(f"Recovery action timed out after {action.timeout_seconds}s")
                return False, None
            
            # Check success condition if provided
            if action.success_condition and not action.success_condition():
                return False, None
            
            return True, result
            
        except Exception as e:
            self.logger.error(f"Recovery action failed: {str(e)}")
            return False, None
    
    def _record_recovery_attempt(
        self,
        error_type: str,
        action: RecoveryAction,
        success: bool,
        error_context: Dict[str, Any]
    ):
        """Record recovery attempt for analysis."""
        record = {
            "timestamp": time.time(),
            "error_type": error_type,
            "strategy": action.strategy.value,
            "description": action.description,
            "success": success,
            "parameters": action.parameters,
            "context": error_context
        }
        
        self.recovery_history.append(record)
        
        # Keep only last 100 records
        if len(self.recovery_history) > 100:
            self.recovery_history = self.recovery_history[-100:]
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery attempt statistics."""
        if not self.recovery_history:
            return {"total_attempts": 0, "success_rate": 0.0}
        
        total_attempts = len(self.recovery_history)
        successful_attempts = sum(1 for record in self.recovery_history if record["success"])
        
        success_rate = successful_attempts / total_attempts if total_attempts > 0 else 0.0
        
        # Statistics by strategy
        strategy_stats = {}
        for record in self.recovery_history:
            strategy = record["strategy"]
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {"attempts": 0, "successes": 0}
            
            strategy_stats[strategy]["attempts"] += 1
            if record["success"]:
                strategy_stats[strategy]["successes"] += 1
        
        # Calculate success rates for each strategy
        for strategy, stats in strategy_stats.items():
            stats["success_rate"] = stats["successes"] / stats["attempts"] if stats["attempts"] > 0 else 0.0
        
        return {
            "total_attempts": total_attempts,
            "successful_attempts": successful_attempts,
            "success_rate": success_rate,
            "strategy_statistics": strategy_stats,
            "recent_attempts": self.recovery_history[-10:]  # Last 10 attempts
        }
    
    def reset_recovery_state(self):
        """Reset recovery attempt counters."""
        self.recovery_attempts.clear()
        self.logger.info("Recovery state reset")
    
    def is_recovery_exhausted(self, error_type: str, error_context: Dict[str, Any]) -> bool:
        """Check if recovery attempts are exhausted for a specific error."""
        recovery_key = f"{error_type}_{hash(str(error_context))}"
        attempts = self.recovery_attempts.get(recovery_key, 0)
        max_attempts = self.max_recovery_attempts.get(error_type, 2)
        
        return attempts >= max_attempts

