"""
ORTools configuration, parameters, and validation.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any


class SolverType(Enum):
    CP_SAT = "cp_sat"
    LINEAR = "linear"
    SAT = "sat"
    SEARCH = "search"


class SolverStatus(Enum):
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class SolverParameters:
    # Global
    log_level: str = "INFO"
    log_console: bool = True
    log_file: bool = True
    time_limit_seconds: Optional[float] = None

    # CP-SAT
    cp_sat_num_search_workers: int = 1
    cp_sat_log_search_progress: bool = True
    cp_sat_linearization_level: int = 1
    cp_sat_symmetry_level: int = 2

    # Linear Solver
    linear_solver_backend: str = "SCIP"

    # Output
    generate_csv: bool = True
    generate_parquet: bool = True
    generate_json: bool = True

    # Objective weights
    weight_time_preference: float = 1.0
    weight_course_preference: float = 1.0
    weight_workload_balance: float = 1.0
    weight_schedule_density: float = 1.0


@dataclass
class ORToolsConfig:
    stage3_output_path: Path
    output_path: Path
    log_path: Path
    error_report_path: Path
    solver_params: SolverParameters = field(default_factory=SolverParameters)
    override_params: Optional[Dict[str, Any]] = None


class ConfigValidator:
    @staticmethod
    def validate(config: ORToolsConfig):
        errors = []
        for p in [config.stage3_output_path, config.output_path, config.log_path, config.error_report_path]:
            if not Path(p).exists() and p in [config.output_path, config.log_path, config.error_report_path]:
                try:
                    Path(p).mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"Failed to create path {p}: {e}")
        return (len(errors) == 0, errors)
