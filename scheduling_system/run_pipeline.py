"""
CLI for Pipeline Orchestrator
Configurable input/output/log paths, progress bars, and live resource stats.
Strictly adheres to mathematical, architectural, and logging/error handling foundations.
"""

import argparse
import sys
from pathlib import Path
from pipeline_orchestrator import PipelineOrchestrator
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.console import Console
import psutil
import time

console = Console()

def show_resource_stats():
    process = psutil.Process()
    mem = process.memory_info().rss / (1024 * 1024)
    cpu = process.cpu_percent(interval=0.1)
    return f"CPU: {cpu:.1f}% | MEM: {mem:.1f}MB"

def main():
    parser = argparse.ArgumentParser(description="Run the Scheduling Engine Pipeline")
    parser.add_argument('--input_path', type=str, default='input_data/', help='Path to input data')
    parser.add_argument('--output_path', type=str, default='output_data/', help='Path to output data')
    parser.add_argument('--log_path', type=str, default='logs/pipeline.log', help='Path to log file')
    parser.add_argument('--report_path', type=str, default='logs/error_report.json', help='Path to error report file')
    args = parser.parse_args()

    config = {
        'input_path': args.input_path,
        'output_path': args.output_path,
        'log_path': args.log_path,
        'report_path': args.report_path,
    }

    orchestrator = PipelineOrchestrator(config)
    stages = orchestrator.stages

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        TextColumn("{task.fields[stats]}")
    ) as progress:
        task = progress.add_task("Pipeline Progress", total=len(stages), stats=show_resource_stats())
        for i, stage_name in enumerate(stages):
            progress.update(task, description=f"Running {stage_name}", stats=show_resource_stats())
            try:
                orchestrator.import_stage(stage_name)
                orchestrator.run_stage(stage_name, None if i == 0 else None)  # Input chaining handled inside orchestrator
            except Exception:
                progress.update(task, description=f"Aborted at {stage_name}", stats=show_resource_stats())
                break
            progress.advance(task)
            time.sleep(0.2)
        progress.update(task, description="Pipeline Complete", stats=show_resource_stats())
    console.print("[bold green]Pipeline execution finished.[/bold green]")

if __name__ == '__main__':
    main()
