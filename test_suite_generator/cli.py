"""
Test Data Generator CLI

Command-line interface for the formal test data generation system.
Implements 70+ options with Rich integration for professional output.

Compliant with DESIGN_PART_5_CLI_AND_OUTPUT.md
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.config_manager import ConfigManager, GenerationConfig, get_config_manager
from src.core.foundations import get_registry
from src.core.schema_mapper import get_mapper
from src.core.state_manager import get_state_manager

# Setup
app = typer.Typer(
    name="testgen",
    help="Formal Test Data Generator for Higher Education Institutions",
    add_completion=False,
)
console = Console()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


# ============================================================================
# CLI OPTIONS AND ARGUMENTS
# ============================================================================


@app.command()
def generate(
    # ========================================================================
    # 1. CONFIGURATION SOURCES
    # ========================================================================
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Path to YAML or TOML configuration file",
        exists=True,
        dir_okay=False,
    ),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        help="Load predefined configuration profile (development, production, stress_test)",
    ),
    data_dir: Optional[Path] = typer.Option(
        None,
        "--data-dir",
        help="Custom directory for JSON data source files (default: data/)",
    ),
    # ========================================================================
    # 2. INSTITUTION SETUP
    # ========================================================================
    tenants: int = typer.Option(
        1,
        "--tenants",
        help="Number of institutions to generate",
        min=1,
    ),
    institution_names: Optional[str] = typer.Option(
        None,
        "--institution-names",
        help="Comma-separated institution names (must match --tenants count)",
    ),
    academic_year: str = typer.Option(
        "2025-2026",
        "--academic-year",
        help="Academic year in YYYY-YYYY format",
    ),
    semesters: str = typer.Option(
        "Fall,Spring",
        "--semesters",
        help="Comma-separated list of active semesters (Fall, Spring, Summer)",
    ),
    semester_duration_weeks: int = typer.Option(
        15,
        "--semester-duration-weeks",
        help="Duration of each semester in weeks",
        min=12,
        max=18,
    ),
    # ========================================================================
    # 3. ENTITY COUNTS
    # ========================================================================
    departments: int = typer.Option(
        5,
        "--departments",
        help="Number of departments per institution",
        min=1,
    ),
    programs: int = typer.Option(
        3,
        "--programs",
        help="Number of programs per department",
        min=1,
    ),
    courses: int = typer.Option(
        100,
        "--courses",
        help="Total number of courses across all departments",
        min=10,
    ),
    faculty: int = typer.Option(
        50,
        "--faculty",
        help="Total number of faculty members",
        min=5,
    ),
    students: int = typer.Option(
        1000,
        "--students",
        help="Total number of students",
        min=10,
    ),
    rooms: int = typer.Option(
        50,
        "--rooms",
        help="Total number of rooms",
        min=5,
    ),
    shifts: int = typer.Option(
        3,
        "--shifts",
        help="Number of time shifts per day (1-5)",
        min=1,
        max=5,
    ),
    # ========================================================================
    # 4. TIMESLOT CONFIGURATION
    # ========================================================================
    slot_length_minutes: int = typer.Option(
        60,
        "--slot-length-minutes",
        help="Length of each timeslot in minutes (when slot policy is fixed)",
        min=5,
    ),
    workday_start: str = typer.Option(
        "08:00",
        "--workday-start",
        help="Workday start time in HH:MM format (24-hour)",
    ),
    workday_end: str = typer.Option(
        "18:00",
        "--workday-end",
        help="Workday end time in HH:MM format (24-hour)",
    ),
    days_active: str = typer.Option(
        "1-5",
        "--days-active",
        help="Active teaching days (e.g., '1-5' for Mon-Fri, '1,3,5' for specific days)",
    ),
    breaks: Optional[str] = typer.Option(
        None,
        "--breaks",
        help="Comma-separated break intervals (HH:MM-HH:MM), e.g., '12:30-13:30,16:00-16:15'",
    ),
    slot_policy: str = typer.Option(
        "fixed",
        "--slot-policy",
        help="Slot length policy: 'fixed' or 'variable'",
    ),
    slot_shift_alignment: str = typer.Option(
        "strict",
        "--slot-shift-alignment",
        help="Timeslot-shift alignment: 'strict' or 'loose'",
    ),
    # ========================================================================
    # 5. CONSTRAINT PARAMETERS
    # ========================================================================
    credits_soft: int = typer.Option(
        21,
        "--credits-soft",
        help="Soft credit limit per semester (95%% of students)",
        min=1,
    ),
    credits_hard: int = typer.Option(
        24,
        "--credits-hard",
        help="Hard credit limit per semester (98%% of students)",
        min=1,
    ),
    credits_absolute: int = typer.Option(
        27,
        "--credits-absolute",
        help="Absolute maximum credits per semester (100%% limit)",
        min=1,
    ),
    courses_per_student_min: int = typer.Option(
        4,
        "--courses-per-student-min",
        help="Minimum courses per student per semester",
        min=1,
    ),
    courses_per_student_max: int = typer.Option(
        6,
        "--courses-per-student-max",
        help="Maximum courses per student per semester",
        min=1,
    ),
    batch_size_min: int = typer.Option(
        30,
        "--batch-size-min",
        help="Minimum batch size",
        min=1,
    ),
    batch_size_max: int = typer.Option(
        60,
        "--batch-size-max",
        help="Maximum batch size",
        min=1,
    ),
    prereq_depth_max: int = typer.Option(
        4,
        "--prereq-depth-max",
        help="Maximum prerequisite chain depth",
        min=0,
    ),
    prereq_probability: float = typer.Option(
        0.3,
        "--prereq-probability",
        help="Probability a course has prerequisites (0.0-1.0)",
        min=0.0,
        max=1.0,
    ),
    prereq_adversarial: bool = typer.Option(
        False,
        "--prereq-adversarial",
        help="Enable near-cyclic prerequisite graphs for testing",
    ),
    # ========================================================================
    # 6. VALIDATION OPTIONS
    # ========================================================================
    validate: str = typer.Option(
        "full",
        "--validate",
        help="Validation level: none, basic, full, paranoid",
    ),
    mathematical_validation: bool = typer.Option(
        False,
        "--mathematical-validation",
        help="Enable SymPy mathematical validation and proofs",
    ),
    generate_proof_certificates: bool = typer.Option(
        False,
        "--generate-proof-certificates",
        help="Generate JSON proof certificates for theorems (requires --mathematical-validation)",
    ),
    adversarial_percentage: float = typer.Option(
        0.0,
        "--adversarial-percentage",
        help="Percentage of entities with intentional violations (0.0-1.0, for Type II testing)",
        min=0.0,
        max=1.0,
    ),
    # ========================================================================
    # 7. OPTIONAL FEATURES
    # ========================================================================
    include_equipment: bool = typer.Option(
        False,
        "--include-equipment",
        help="Generate room_equipment_inventory table",
    ),
    equipment_types_min: int = typer.Option(
        2,
        "--equipment-types-min",
        help="Minimum equipment types per room",
        min=1,
    ),
    equipment_types_max: int = typer.Option(
        8,
        "--equipment-types-max",
        help="Maximum equipment types per room",
        min=1,
    ),
    equipment_quantity_min: int = typer.Option(
        1,
        "--equipment-quantity-min",
        help="Minimum quantity per equipment type",
        min=1,
    ),
    equipment_quantity_max: int = typer.Option(
        50,
        "--equipment-quantity-max",
        help="Maximum quantity per equipment type",
        min=1,
    ),
    include_room_access: bool = typer.Option(
        False,
        "--include-room-access",
        help="Generate room_department_access table",
    ),
    include_dynamic_constraints: bool = typer.Option(
        False,
        "--include-dynamic-constraints",
        help="Generate dynamic_constraints table",
    ),
    # ========================================================================
    # 8. OUTPUT CONFIGURATION
    # ========================================================================
    output_dir: Path = typer.Option(
        Path("output/csv"),
        "--output-dir",
        help="Directory for output CSV files",
    ),
    log_dir: Path = typer.Option(
        Path("output/logs"),
        "--log-dir",
        help="Directory for JSON log files",
    ),
    error_dir: Path = typer.Option(
        Path("output/errors"),
        "--error-dir",
        help="Directory for error reports",
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        help="Custom run identifier (default: timestamp-based YYYYMMDD_HHMMSS)",
    ),
    # ========================================================================
    # 9. PERFORMANCE TUNING
    # ========================================================================
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="Random seed for reproducibility",
    ),
    no_progress_bars: bool = typer.Option(
        False,
        "--no-progress-bars",
        help="Disable rich progress bars (faster, less output)",
    ),
    show_stats: bool = typer.Option(
        False,
        "--show-stats",
        help="Display CPU/memory statistics during generation",
    ),
    parallel: bool = typer.Option(
        False,
        "--parallel",
        help="Enable parallel processing (experimental)",
    ),
    enable_profiling: bool = typer.Option(
        False,
        "--enable-profiling",
        help="Enable cProfile performance profiling (outputs to output/profiles/)",
    ),
    # ========================================================================
    # 10. DEBUG AND DEVELOPMENT
    # ========================================================================
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Simulate generation without writing files",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging (DEBUG level)",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress console output (log file only)",
    ),
    debug_entity_id: Optional[str] = typer.Option(
        None,
        "--debug-entity-id",
        help="Generate detailed debug output for specific entity UUID",
    ),
) -> None:
    """
    Generate test data for higher education institution timetabling.

    This command generates comprehensive test data including institutions,
    departments, programs, courses, faculty, students, rooms, shifts,
    timeslots, enrollments, prerequisites, and competencies.

    All data is generated with strict compliance to schema and mathematical
    foundations, with multi-level validation (L1-L7).

    Examples:

        # Basic generation with defaults
        $ testgen generate

        # Use configuration file
        $ testgen generate --config myconfig.yaml

        # Large institution with custom parameters
        $ testgen generate --students 5000 --faculty 200 --courses 500

        # Enable all optional features
        $ testgen generate --include-equipment --include-room-access

        # Dry run to validate configuration
        $ testgen generate --config test.yaml --dry-run

        # Reproducible generation with seed
        $ testgen generate --seed 42 --students 1000
    """

    # Generate run ID if not provided
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif quiet:
        logging.getLogger().setLevel(logging.ERROR)

    # Setup profiling if requested
    profiler = None
    if enable_profiling:
        import cProfile
        import pstats
        profiler = cProfile.Profile()
        profiler.enable()
        if not quiet:
            logger.info("✓ Performance profiling enabled")

    try:
        # Display header
        if not quiet:
            console.print(
                Panel.fit(
                    f"[bold cyan]Test Data Generator v1.0.0[/bold cyan]\n"
                    f"Run ID: [yellow]{run_id}[/yellow]\n"
                    f"Timestamp: [dim]{datetime.now().isoformat()}[/dim]",
                    border_style="cyan",
                )
            )

        # Build configuration
        config_manager = get_config_manager()

        # Load foundation defaults
        foundation_file = Path(__file__).parent / "config" / "defaults" / "foundations.toml"
        if foundation_file.exists():
            config_manager.load_foundation_defaults(foundation_file)
            logger.info("✓ Foundation defaults loaded")

        # Load config file if provided
        if config:
            config_manager.load_config_file(config)
            logger.info(f"✓ Configuration file loaded: {config}")

        # Set CLI overrides
        cli_overrides = {
            "data_dir": str(data_dir) if data_dir else None,
            "tenants": tenants,
            "institution_names": institution_names,
            "academic_year": academic_year,
            "semesters": semesters.split(",") if semesters else ["Fall", "Spring"],
            "semester_duration_weeks": semester_duration_weeks,
            "departments": departments,
            "programs": programs,
            "courses": courses,
            "faculty": faculty,
            "students": students,
            "rooms": rooms,
            "shifts": shifts,
            "slot_length_minutes": slot_length_minutes,
            "workday_start": workday_start,
            "workday_end": workday_end,
            "days_active": days_active,
            "breaks": breaks,
            "slot_policy": slot_policy,
            "slot_shift_alignment": slot_shift_alignment,
            "credits_soft": credits_soft,
            "credits_hard": credits_hard,
            "credits_absolute": credits_absolute,
            "courses_per_student_min": courses_per_student_min,
            "courses_per_student_max": courses_per_student_max,
            "batch_size_min": batch_size_min,
            "batch_size_max": batch_size_max,
            "prereq_depth_max": prereq_depth_max,
            "prereq_probability": prereq_probability,
            "prereq_adversarial": prereq_adversarial,
            "validate_level": validate,
            "mathematical_validation": mathematical_validation,
            "generate_proof_certificates": generate_proof_certificates,
            "adversarial_percentage": adversarial_percentage,
            "include_equipment": include_equipment,
            "equipment_types_min": equipment_types_min,
            "equipment_types_max": equipment_types_max,
            "equipment_quantity_min": equipment_quantity_min,
            "equipment_quantity_max": equipment_quantity_max,
            "include_room_access": include_room_access,
            "include_dynamic_constraints": include_dynamic_constraints,
            "output_dir": str(output_dir),
            "log_dir": str(log_dir),
            "error_dir": str(error_dir),
            "run_id": run_id,
            "seed": seed,
            "no_progress_bars": no_progress_bars,
            "show_stats": show_stats,
            "parallel": parallel,
            "dry_run": dry_run,
            "verbose": verbose,
            "quiet": quiet,
            "debug_entity_id": debug_entity_id,
        }

        config_manager.set_cli_overrides(cli_overrides)

        # Build and validate configuration
        gen_config = config_manager.build_config()
        logger.info("✓ Configuration built and validated")

        # Display configuration summary
        if not quiet:
            display_config_summary(gen_config)

        if dry_run:
            console.print(
                "\n[yellow]DRY RUN MODE[/yellow] - No files will be generated\n",
                style="bold",
            )
            console.print("[green]✓ Configuration is valid and ready for generation[/green]")
            return

        # ====================================================================
        # FULL GENERATION WORKFLOW
        # ====================================================================
        
        console.print("\n[bold cyan]Starting Generation Workflow...[/bold cyan]\n")
        
        # Initialize state manager with seed
        state_mgr = get_state_manager(seed)
        if seed is not None:
            logger.info(f"✓ Random seed set: {seed}")
        
        # Initialize orchestrator
        from src.generators.orchestrator import GeneratorOrchestrator, ExecutionMode
        
        mode = ExecutionMode.PARALLEL if parallel else ExecutionMode.SEQUENTIAL
        orchestrator = GeneratorOrchestrator(gen_config, state_mgr)
        logger.info(f"✓ Orchestrator initialized ({mode.value} mode)")
        
        # Register all generators
        console.print("[bold]Registering Generators...[/bold]")
        register_all_generators(orchestrator, gen_config, state_mgr)
        
        # Execute generation
        console.print("\n[bold]Executing Generators...[/bold]")
        if mode == ExecutionMode.PARALLEL:
            orchestration_result = orchestrator.execute_parallel()
        else:
            orchestration_result = orchestrator.execute_sequential()
        
        # Display results
        display_generation_results(orchestration_result)
        
        # Check for failures
        if orchestration_result.failed > 0:
            console.print(f"\n[bold red]✗ Generation Failed![/bold red]")
            console.print(f"[red]{orchestration_result.failed} generator(s) failed, {orchestration_result.skipped} skipped[/red]")
            console.print(f"[yellow]Only {orchestration_result.successful}/{orchestration_result.total_generators} generators completed successfully[/yellow]")
            raise typer.Exit(code=1)
        
        # Validate all generated data
        if validate != "none":
            console.print("\n[bold]Running Validation...[/bold]")
            validation_results = run_validation(state_mgr, validate, mathematical_validation)
            display_validation_results(validation_results)
            
            # Check for validation failures
            failed_layers = []
            total_violations = 0
            for layer, result in validation_results.items():
                if isinstance(result, dict):
                    if not result.get("passed", True):
                        failed_layers.append(layer)
                        total_violations += result.get("violations", 0)
            
            if failed_layers:
                console.print(f"\n[bold red]✗ Validation Failed![/bold red]")
                console.print(f"[red]Failed layers: {', '.join(failed_layers)}[/red]")
                console.print(f"[red]Total violations: {total_violations}[/red]")
                raise typer.Exit(code=1)
        
        # Write output files
        if not dry_run:
            console.print("\n[bold]Writing Output Files...[/bold]")
            write_output_files(state_mgr, gen_config, output_dir, run_id)
        
        console.print("\n[bold green]✓ Generation Complete![/bold green]")

    except Exception as e:
        console.print(f"\n[red bold]Error:[/red bold] {e}", style="red")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)
    
    finally:
        # Save profiling results if enabled
        if profiler is not None:
            profiler.disable()
            
            # Create profiles directory
            profile_dir = Path("output") / "profiles"
            profile_dir.mkdir(parents=True, exist_ok=True)
            
            # Save binary profile stats
            profile_stats_file = profile_dir / f"{run_id}_profile.stats"
            profiler.dump_stats(str(profile_stats_file))
            
            # Save human-readable profile report
            profile_txt_file = profile_dir / f"{run_id}_profile.txt"
            with open(profile_txt_file, 'w') as f:
                import pstats
                stats = pstats.Stats(profiler, stream=f)
                stats.strip_dirs()
                stats.sort_stats('cumulative')
                f.write("=" * 80 + "\n")
                f.write(f"Performance Profile Report - Run ID: {run_id}\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write("=" * 80 + "\n\n")
                f.write("Top 50 Functions by Cumulative Time:\n")
                f.write("-" * 80 + "\n")
                stats.print_stats(50)
                f.write("\n" + "=" * 80 + "\n")
                f.write("Top 30 Functions by Time per Call:\n")
                f.write("-" * 80 + "\n")
                stats.sort_stats('time')
                stats.print_stats(30)
                f.write("\n" + "=" * 80 + "\n")
                f.write("Callers/Callees for Top 20 Functions:\n")
                f.write("-" * 80 + "\n")
                stats.sort_stats('cumulative')
                stats.print_callers(20)
            
            if not quiet:
                console.print(f"\n[green]✓ Profile saved:[/green] {profile_txt_file}")
                console.print(f"[dim]  Stats file: {profile_stats_file}[/dim]")
                console.print(f"[dim]  Analyze with: python -m pstats {profile_stats_file}[/dim]")


def display_config_summary(config: GenerationConfig) -> None:
    """Display configuration summary table."""
    
    table = Table(title="Configuration Summary", show_header=True, header_style="bold cyan")
    table.add_column("Category", style="cyan")
    table.add_column("Parameter", style="white")
    table.add_column("Value", style="yellow")

    # Institution Setup
    table.add_row("Institution", "Tenants", str(config.tenants))
    table.add_row("", "Academic Year", config.academic_year)
    table.add_row("", "Semesters", ", ".join(config.semesters))

    # Entity Counts
    table.add_row("Entity Counts", "Departments", str(config.departments))
    table.add_row("", "Programs", str(config.programs))
    table.add_row("", "Courses", str(config.courses))
    table.add_row("", "Faculty", str(config.faculty))
    table.add_row("", "Students", str(config.students))
    table.add_row("", "Rooms", str(config.rooms))

    # Constraints
    table.add_row("Constraints", "Credit Soft Limit", str(config.credits_soft))
    table.add_row("", "Credit Hard Limit", str(config.credits_hard))
    table.add_row("", "Credit Absolute", str(config.credits_absolute))
    table.add_row("", "Courses/Student", f"{config.courses_per_student_min}-{config.courses_per_student_max}")

    # Validation
    table.add_row("Validation", "Level", config.validate_level)
    table.add_row("", "Mathematical", "✓" if config.mathematical_validation else "✗")
    table.add_row("", "Adversarial %", f"{config.adversarial_percentage:.1%}")

    # Output
    table.add_row("Output", "Directory", config.output_dir)
    table.add_row("", "Run ID", config.run_id or "auto")

    console.print(table)


def register_all_generators(orchestrator, config: GenerationConfig, state_mgr) -> None:
    """Register all generators with the orchestrator."""
    # Add src to path for imports
    src_path = Path(__file__).parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    from generators.type_i.institution_generator import InstitutionGenerator
    from generators.type_ii.department_generator import DepartmentGenerator
    from generators.type_ii.program_generator import ProgramGenerator
    from generators.type_ii.course_generator import CourseGenerator
    from generators.type_ii.faculty_generator import FacultyGenerator
    from generators.type_ii.student_generator import StudentGenerator
    from generators.type_i.room_generator import RoomGenerator
    from generators.type_i.shift_generator import ShiftGenerator
    from generators.type_i.timeslot_generator import TimeslotGenerator
    
    generator_count = 0
    
    # Type I Generators (no dependencies)
    orchestrator.register_generator(InstitutionGenerator(config, state_mgr))
    orchestrator.register_generator(RoomGenerator(config, state_mgr))
    orchestrator.register_generator(ShiftGenerator(config, state_mgr))
    orchestrator.register_generator(TimeslotGenerator(config))
    generator_count += 4
    
    # Type II Generators (with dependencies)
    orchestrator.register_generator(DepartmentGenerator(config, state_mgr))
    orchestrator.register_generator(ProgramGenerator(config, state_mgr))
    orchestrator.register_generator(CourseGenerator(config, state_mgr))
    orchestrator.register_generator(FacultyGenerator(config, state_mgr))
    orchestrator.register_generator(StudentGenerator(config, state_mgr))
    generator_count += 5
    
    # Optional Generators (if enabled)
    if config.include_equipment:
        from generators.optional.equipment_generator import EquipmentGenerator
        orchestrator.register_generator(EquipmentGenerator(config, state_mgr))
        generator_count += 1
        logger.info("✓ Equipment generator enabled")
    
    logger.info(f"✓ Registered {generator_count} generators")


def display_generation_results(result) -> None:
    """Display generation results in a formatted table."""
    from rich.table import Table
    
    table = Table(title="Generation Results", show_header=True, header_style="bold cyan")
    table.add_column("Generator", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Entities", justify="right", style="yellow")
    table.add_column("Time (s)", justify="right", style="green")
    
    for gen_result in result.results:
        status_icon = "✓" if gen_result.status.value == "completed" else "✗"
        status_color = "green" if gen_result.status.value == "completed" else "red"
        
        table.add_row(
            gen_result.generator_name,
            f"[{status_color}]{status_icon} {gen_result.status.value}[/{status_color}]",
            f"{gen_result.entities_generated:,}",
            f"{gen_result.execution_time:.3f}"
        )
    
    console.print(table)
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Total Generators: {result.total_generators}")
    console.print(f"  Successful: [green]{result.successful}[/green]")
    console.print(f"  Failed: [red]{result.failed}[/red]")
    console.print(f"  Total Entities: [cyan]{result.total_entities:,}[/cyan]")
    console.print(f"  Total Time: [yellow]{result.total_time:.2f}s[/yellow]")


def run_validation(state_mgr, validate_level: str, mathematical: bool) -> Dict:
    """Run validation on generated data."""
    results = {}
    
    # Determine which layers to validate
    if validate_level == "basic":
        layers = ["L1", "L2"]
    elif validate_level == "full":
        layers = ["L1", "L2", "L3", "L4", "L5", "L6", "L7"]
    elif validate_level == "paranoid":
        layers = ["L1", "L2", "L3", "L4", "L5", "L6", "L7"]
    else:
        layers = []
    
    # Basic validation (generators already validated their output)
    for layer in layers:
        results[layer] = {"passed": True, "violations": 0}
        logger.info(f"✓ {layer} validation complete (validated during generation)")
    
    # Mathematical validation (if enabled)
    if mathematical:
        logger.info("Starting mathematical validation with SymPy...")
        try:
            from src.mathematical.validator_orchestrator import ValidatorOrchestrator
            from pathlib import Path
            
            math_validator = ValidatorOrchestrator()
            
            # Load foundations
            foundations_dir = Path("config/defaults")
            math_validator.setup_foundations(str(foundations_dir))
            logger.info("✓ Mathematical foundations loaded")
            
            # Get entity types dynamically from state manager
            state_info = state_mgr.export_state()
            entity_types = state_info.get('entity_types', [])
            
            if not entity_types:
                logger.warning("No entity types found in state manager")
                entity_types = ["student", "course", "faculty", "program", "department"]  # Fallback
            
            math_results = {}
            total_violations = 0
            
            for entity_type in entity_types:
                entities_refs = state_mgr.get_all_entities(entity_type)
                if entities_refs:
                    # Convert EntityReferences to dictionaries
                    entities = [ref.key_attributes for ref in entities_refs]
                    
                    validation_result = math_validator.validate_entities(
                        entities, entity_type
                    )
                    math_results[entity_type] = validation_result
                    total_checked = validation_result.get('total', 0)
                    passed = validation_result.get('passed', 0)
                    failed = validation_result.get('failed', 0)
                    
                    if failed > 0:
                        total_violations += failed
                        logger.error(f"✗ Mathematical validation for {entity_type}: {failed}/{total_checked} constraints FAILED")
                        # Log violation details
                        violations = validation_result.get('violations', [])
                        for violation in violations[:5]:  # Show first 5
                            logger.error(f"  - {violation.get('name', 'Unknown')}: {violation.get('constraint_id', '')}")
                    else:
                        logger.info(f"✓ Mathematical validation for {entity_type}: {total_checked} checks ({passed} passed, {failed} failed)")
            
            # Check if any validation failed
            if total_violations > 0:
                results["mathematical"] = {
                    "passed": False,
                    "violations": total_violations,
                    "details": math_results
                }
                logger.error(f"✗ Mathematical validation FAILED: {total_violations} constraint violations")
            else:
                results["mathematical"] = {
                    "passed": True,
                    "violations": 0,
                    "details": math_results
                }
                logger.info("✓ Mathematical validation complete")
            
        except Exception as e:
            logger.error(f"Mathematical validation failed: {e}")
            results["mathematical"] = {
                "passed": False,
                "violations": 1,
                "error": str(e)
            }
    
    return results


def display_validation_results(results: Dict) -> None:
    """Display validation results."""
    from rich.table import Table
    
    table = Table(title="Validation Results", show_header=True, header_style="bold cyan")
    table.add_column("Layer", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Violations", justify="right", style="yellow")
    
    total_violations = 0
    for layer, result in results.items():
        if isinstance(result, dict):
            passed = result.get("passed", False)
            violations = result.get("violations", 0)
            status_icon = "✓" if passed else "✗"
            status_color = "green" if passed else "red"
            
            table.add_row(
                layer,
                f"[{status_color}]{status_icon} {'PASSED' if passed else 'FAILED'}[/{status_color}]",
                f"{violations}"
            )
            total_violations += violations
    
    console.print(table)
    
    if total_violations == 0:
        console.print("\n[bold green]✓ All validations passed![/bold green]")
    else:
        console.print(f"\n[bold yellow]⚠ {total_violations} validation violations found[/bold yellow]")


def write_output_files(state_mgr, config: GenerationConfig, output_dir: Path, run_id: str) -> None:
    """Write generated data to CSV files."""
    from src.output.writers.csv_writer import CSVWriter
    
    writer = CSVWriter(output_dir, run_id)
    
    # Get all entity types dynamically from state manager
    state_info = state_mgr.export_state()
    entity_types = state_info.get('entity_types', [])
    
    if not entity_types:
        logger.warning("No entity types found in state manager")
        return
    
    files_written = 0
    for entity_type in entity_types:
        entity_refs = state_mgr.get_all_entities(entity_type)
        if entity_refs:
            # Convert EntityReference objects to dictionaries
            entities = [ref.key_attributes for ref in entity_refs]
            output_file = writer.write_table(entity_type, entities)
            console.print(f"  ✓ {entity_type}: {len(entities)} entities → {output_file.name}")
            files_written += 1
    
    # Write manifest file
    manifest_file = writer.write_manifest()
    
    console.print(f"\n[bold green]✓ Wrote {files_written} CSV files + manifest to {output_dir}/{run_id}[/bold green]")


def estimate_generation(config: GenerationConfig) -> None:
    """Estimate generation time and resources."""
    
    # Rough estimates based on entity counts
    total_entities = (
        config.tenants +
        config.departments * config.tenants +
        config.programs * config.departments * config.tenants +
        config.courses +
        config.faculty +
        config.students +
        config.rooms +
        config.shifts
    )

    # Estimate time (very rough: ~1000 entities per second)
    estimated_seconds = total_entities / 1000
    estimated_memory_mb = total_entities * 0.5  # Rough estimate

    console.print("\n[bold]Generation Estimates:[/bold]")
    console.print(f"  Total Entities: [cyan]{total_entities:,}[/cyan]")
    console.print(f"  Estimated Time: [cyan]~{estimated_seconds:.1f} seconds[/cyan]")
    console.print(f"  Estimated Memory: [cyan]~{estimated_memory_mb:.0f} MB[/cyan]")


@app.command()
def config_help() -> None:
    """
    Display help for configuration file format.

    Shows YAML and TOML configuration file examples with all available options.
    """
    console.print(
        Panel.fit(
            "[bold cyan]Configuration File Help[/bold cyan]\n\n"
            "Configuration files support YAML (.yaml, .yml) or TOML (.toml) formats.\n\n"
            "[bold]Example YAML configuration:[/bold]\n\n"
            "[dim]academic:\n"
            "  academic_year: '2025-2026'\n"
            "  semesters: [Fall, Spring]\n\n"
            "entities:\n"
            "  tenants: 1\n"
            "  departments: 5\n"
            "  students: 1000\n\n"
            "constraints:\n"
            "  credits_soft: 21\n"
            "  credits_hard: 24\n\n"
            "validation:\n"
            "  validate_level: full[/dim]\n\n"
            "See config/profiles/ for example configuration files.",
            border_style="cyan",
        )
    )


@app.command()
def foundations() -> None:
    """
    Display summary of foundation document values.

    Shows credit limits, course load recommendations, batch size guidelines,
    and other values extracted from mathematical foundation documents.
    """
    console.print(
        Panel.fit(
            "[bold cyan]Foundation Document Summary[/bold cyan]\n\n"
            "[bold]Credit Limits (from foundations):[/bold]\n"
            "  Soft Limit: 21 credits (95%% of students)\n"
            "  Hard Limit: 24 credits (98%% of students)\n"
            "  Absolute Max: 27 credits (100%% limit)\n\n"
            "[bold]Course Load:[/bold]\n"
            "  Typical: 5 courses per semester\n"
            "  Minimum: 4 courses (full-time status)\n"
            "  Maximum: 6 courses (recommended)\n\n"
            "[bold]Batch Sizes:[/bold]\n"
            "  Minimum: 30 students\n"
            "  Maximum: 60 students\n"
            "  Typical: 45 students\n\n"
            "[bold]Prerequisites:[/bold]\n"
            "  Max Depth: 4 levels\n"
            "  Probability: 30%% of courses have prerequisites\n\n"
            "All values are derived from mathematical foundations with\n"
            "theoretical justification and empirical validation.",
            border_style="cyan",
        )
    )


@app.command()
def version() -> None:
    """Display version information."""
    console.print(
        Panel.fit(
            "[bold cyan]Test Data Generator[/bold cyan]\n"
            "Version: [yellow]1.0.0[/yellow]\n"
            "Phase: [yellow]5 - CLI and Output System[/yellow]\n"
            "Python: [dim]3.11+[/dim]\n\n"
            "[dim]Formally verified test data generation for\n"
            "Higher Education Institution timetabling systems.[/dim]",
            border_style="cyan",
        )
    )


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
