# Test Suite Generator

## Overview

The LUMEN Test Suite Generator is a sophisticated synthetic data generation system designed to create realistic, NEP-2020 compliant test datasets for comprehensive testing and validation of the scheduling engine. The generator produces datasets across varying complexity levels and institutional scenarios.

## Purpose

- **Development Testing**: Generate controlled datasets for feature development
- **Performance Benchmarking**: Test system performance across different problem sizes
- **Edge Case Testing**: Create challenging scenarios to test system robustness
- **Compliance Validation**: Verify NEP-2020 policy compliance in generated schedules
- **Demo Data**: Produce realistic demo data for presentations and training

## Architecture

```
test_suite_generator/
├── cli.py                    # Command-line interface
├── requirements.txt          # Python dependencies
├── requirements-dev.txt      # Development dependencies
│
├── config/                   # Configuration templates
│   ├── small_institution.yaml
│   ├── medium_institution.yaml
│   └── large_institution.yaml
│
├── data/                     # Generated output
│   ├── generated/            # Generated datasets
│   └── templates/            # Data templates
│
└── src/                      # Source code
    ├── __init__.py
    ├── core/                 # Core generation logic
    ├── generators/           # Entity generators
    │   ├── type_i/           # Mandatory entities
    │   │   ├── faculty_generator.py
    │   │   ├── course_generator.py
    │   │   ├── student_generator.py
    │   │   ├── room_generator.py
    │   │   └── timeslot_generator.py
    │   └── type_ii/          # Optional entities
    │       ├── constraint_generator.py
    │       ├── equipment_generator.py
    │       └── prerequisite_generator.py
    ├── validation/           # Data validation
    │   └── layers/           # Multi-layer validation
    └── utils/                # Utility functions
```

## Features

### Institutional Profiles

The generator supports three pre-configured institutional profiles:

#### Small Institution
- **Students**: 500-1,000
- **Faculty**: 30-50
- **Courses**: 50-100
- **Rooms**: 15-25
- **Departments**: 3-5
- **Use Case**: Small colleges, specialized institutions

#### Medium Institution
- **Students**: 1,000-5,000
- **Faculty**: 50-150
- **Courses**: 100-300
- **Rooms**: 25-75
- **Departments**: 5-10
- **Use Case**: Standard universities, engineering colleges

#### Large Institution
- **Students**: 5,000-15,000
- **Faculty**: 150-500
- **Courses**: 300-1,000
- **Rooms**: 75-200
- **Departments**: 10-20
- **Use Case**: Large universities, multi-campus institutions

### Data Entity Types

#### Type I: Mandatory Entities

**1. Faculty**
- Faculty ID, name, department
- Employment type (REGULAR, CONTRACT, VISITING, ADJUNCT)
- Specialization and qualifications
- Maximum weekly teaching hours
- Preferred and unavailable timeslots

**2. Courses**
- Course code and name
- Credits and weekly hours
- Course type (THEORY, LAB, PRACTICAL, ELECTIVE)
- Department and semester
- Prerequisites
- Student capacity limits

**3. Students**
- Student ID, name, roll number
- Department and program
- Current semester and admission year
- Email and contact information

**4. Sections**
- Section ID and name
- Associated course
- Enrolled student count
- Assigned faculty
- Room type requirements

**5. Rooms**
- Room number and building
- Room type (CLASSROOM, LAB, AUDITORIUM, SEMINAR_HALL)
- Capacity
- Facilities (projector, AC, whiteboard)
- Department ownership (optional)

**6. Timeslots**
- Day of week
- Start and end time
- Timeslot type (MORNING, AFTERNOON, EVENING)
- Active status

#### Type II: Optional Entities

**7. Faculty Constraints**
- Unavailability windows
- Preferred teaching times
- Maximum consecutive hours
- Minimum gap requirements

**8. Course Prerequisites**
- Prerequisite relationships
- Co-requisite requirements
- Strict vs. recommended dependencies

**9. Room Equipment**
- Equipment type and quantity
- Operational status
- Special requirements

**10. Room Access Policies**
- Department-specific access
- Time-based restrictions
- Special permission requirements

**11. Holidays and Exceptions**
- Holiday calendar
- Exam periods
- Special events

## Installation

### Requirements

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
cd test_suite_generator
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

### Key Dependencies

- `pandas>=1.3.0`: Data manipulation
- `numpy>=1.21.0`: Random number generation and statistics
- `faker>=8.0.0`: Realistic name and data generation
- `pydantic>=1.8.0`: Data validation
- `pyyaml>=5.4.0`: Configuration file parsing

## Usage

### Command-Line Interface

#### Basic Generation

```bash
# Generate small institution dataset
python -m src.cli generate --size small --output ./data/generated/small

# Generate medium institution dataset
python -m src.cli generate --size medium --output ./data/generated/medium

# Generate large institution dataset
python -m src.cli generate --size large --output ./data/generated/large
```

#### Custom Configuration

```bash
# Use custom configuration file
python -m src.cli generate --config ./config/custom.yaml --output ./data/generated/custom

# Generate with specific seed (for reproducibility)
python -m src.cli generate --size medium --seed 42 --output ./data/generated/reproducible
```

#### Advanced Options

```bash
# Generate Type I entities only
python -m src.cli generate --size medium --type-i-only --output ./data/generated/type_i

# Include Type II optional entities
python -m src.cli generate --size medium --include-type-ii --output ./data/generated/complete

# Set custom complexity parameters
python -m src.cli generate --faculty 100 --students 2000 --courses 200 --output ./data/generated/custom_size

# Enable validation
python -m src.cli generate --size medium --validate --output ./data/generated/validated
```

### Python API

```python
from src.core.generator import TestSuiteGenerator
from src.config.profiles import InstitutionProfile

# Initialize generator
generator = TestSuiteGenerator()

# Generate using profile
profile = InstitutionProfile.MEDIUM
data = generator.generate(profile)

# Save to files
generator.save_csv(data, output_dir="./data/generated")

# Generate with custom parameters
custom_params = {
    "num_students": 1500,
    "num_faculty": 80,
    "num_courses": 150,
    "num_rooms": 40,
    "num_departments": 7
}
data = generator.generate_custom(custom_params)
```

## Configuration

### Configuration File Format (YAML)

```yaml
institution:
  name: "Sample University"
  type: "Engineering"
  
parameters:
  num_departments: 5
  num_faculty: 100
  num_students: 2000
  num_courses: 200
  num_rooms: 50
  num_timeslots: 50

constraints:
  max_faculty_hours: 18
  min_faculty_hours: 12
  max_students_per_section: 60
  min_room_capacity: 30
  max_room_capacity: 120

timeslots:
  working_days: ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY"]
  start_time: "08:00"
  end_time: "17:00"
  slot_duration: 60  # minutes
  break_duration: 10  # minutes

generation_options:
  include_type_ii: true
  nep2020_compliant: true
  realistic_names: true
  seed: null  # null for random, integer for reproducibility
```

### NEP-2020 Compliance Parameters

The generator ensures compliance with NEP-2020 requirements:

- **Faculty Workload**: 12-18 hours per week for regular faculty
- **Student Batch Size**: 30-60 students per section (theory), 15-30 (lab)
- **Credit Structure**: Courses follow standard credit hour definitions
- **Multidisciplinary**: Cross-department course offerings
- **Flexible Curriculum**: Elective and core course balance

## Output Format

### Generated Files

The generator produces CSV files compatible with the scheduling engine:

```
output_directory/
├── faculty.csv               # Faculty information
├── courses.csv               # Course catalog
├── students.csv              # Student enrollment
├── sections.csv              # Course sections
├── rooms.csv                 # Available rooms
├── timeslots.csv             # Available timeslots
├── constraints.csv           # Faculty constraints (Type II)
├── prerequisites.csv         # Course prerequisites (Type II)
├── equipment.csv             # Room equipment (Type II)
├── room_access.csv           # Room access policies (Type II)
└── metadata.json             # Generation metadata
```

### Metadata File

```json
{
  "generation_date": "2025-10-28T10:30:00Z",
  "generator_version": "1.0.0",
  "institution_profile": "MEDIUM",
  "parameters": {
    "num_students": 2000,
    "num_faculty": 100,
    "num_courses": 200,
    "num_rooms": 50
  },
  "statistics": {
    "total_teaching_hours": 1200,
    "avg_faculty_load": 12.0,
    "room_utilization": 0.75
  },
  "compliance": {
    "nep2020_compliant": true,
    "validation_passed": true
  }
}
```

## Data Generation Algorithms

### Faculty Generation

1. **Name Generation**: Realistic names using Faker library
2. **Department Assignment**: Distributed across departments
3. **Specialization**: Random from domain-specific list
4. **Workload**: Within NEP-2020 limits (12-18 hours)
5. **Preferences**: Random timeslot preferences with constraints

### Course Generation

1. **Course Codes**: Department prefix + sequential numbers
2. **Credit Hours**: Standard credit structure (2, 3, 4 credits)
3. **Type Distribution**: 70% theory, 20% lab, 10% elective
4. **Prerequisites**: DAG (Directed Acyclic Graph) to avoid cycles
5. **Capacity**: Based on institutional profile

### Student Generation

1. **Enrollment Distribution**: Balanced across departments and years
2. **Roll Numbers**: Format: YYYY-DEPT-NNNN
3. **Email Generation**: Standardized format: rollnumber@institution.edu
4. **Program Assignment**: Based on department (B.Tech, M.Tech, etc.)

### Section Generation

1. **Section Count**: Based on enrolled students and capacity
2. **Faculty Assignment**: Match faculty specialization to course
3. **Room Requirements**: Map course type to room type
4. **Enrollment**: Realistic distribution within capacity limits

### Room Generation

1. **Room Numbers**: Building prefix + floor + room number
2. **Type Distribution**: 60% classrooms, 25% labs, 10% seminar, 5% auditorium
3. **Capacity**: Varied based on type (30-120 students)
4. **Facilities**: Random equipment based on room type

### Timeslot Generation

1. **Working Hours**: Configurable start and end times
2. **Slot Duration**: Standard 1-hour slots
3. **Break Times**: Lunch break and short breaks
4. **Day Distribution**: Monday-Friday (configurable)

## Validation

### Multi-Layer Validation

The generator includes built-in validation:

**Layer 1: Syntactic Validation**
- CSV format correctness
- Data type validation
- Required field presence

**Layer 2: Domain Validation**
- Value ranges (capacity > 0, hours ≥ 0)
- Enum values (employment types, course types)
- Date/time formats

**Layer 3: Referential Integrity**
- Foreign key relationships
- No orphaned records
- Consistent references

**Layer 4: Business Rules**
- NEP-2020 compliance
- Resource feasibility
- Constraint consistency

### Running Validation

```bash
# Validate generated data
python -m src.cli validate --input ./data/generated/medium

# Validate with detailed report
python -m src.cli validate --input ./data/generated/medium --detailed --output ./validation_report.txt
```

## Testing

### Unit Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test modules
python -m pytest tests/test_faculty_generator.py
python -m pytest tests/test_validation.py

# Run with coverage
python -m pytest --cov=src tests/
```

### Integration Testing

```bash
# Generate and validate in one step
python -m src.cli generate --size medium --validate --output ./test_output

# Use generated data with scheduling engine
cd ../scheduling_system
python run_pipeline.py --input-dir ../test_suite_generator/test_output
```

## Performance

### Generation Times (Approximate)

- **Small Institution**: 2-5 seconds
- **Medium Institution**: 5-15 seconds
- **Large Institution**: 15-45 seconds

### Memory Usage

- **Small**: < 100 MB
- **Medium**: 100-500 MB
- **Large**: 500 MB - 2 GB

## Customization

### Adding Custom Generators

1. Create generator class in `src/generators/`
2. Inherit from `BaseGenerator`
3. Implement `generate()` method
4. Register in generator factory

Example:
```python
from src.generators.base import BaseGenerator

class CustomEntityGenerator(BaseGenerator):
    def generate(self, count: int, params: dict) -> pd.DataFrame:
        # Generation logic
        return dataframe
```

### Custom Validation Rules

1. Create validator in `src/validation/`
2. Inherit from `BaseValidator`
3. Implement `validate()` method
4. Add to validation pipeline

## Troubleshooting

### Common Issues

**Import Errors**:
```bash
pip install -r requirements.txt
```

**Invalid Configuration**:
```bash
# Validate config file
python -m src.cli validate-config --config ./config/custom.yaml
```

**Generation Failures**:
```bash
# Enable debug logging
python -m src.cli generate --size medium --log-level DEBUG --output ./data/generated
```

## Examples

### Reproducible Dataset

```bash
# Same seed produces identical output
python -m src.cli generate --size medium --seed 12345 --output ./data/reproducible
```

### Minimal Dataset (Quick Testing)

```bash
python -m src.cli generate --faculty 10 --students 100 --courses 20 --output ./data/minimal
```

### Complex Dataset (Stress Testing)

```bash
python -m src.cli generate --size large --include-type-ii --output ./data/complex
```

## Documentation

- **Technical Details**: See source code docstrings
- **Algorithm Descriptions**: See `src/generators/` module documentation
- **Validation Framework**: See `src/validation/` documentation

## Support

- **System Documentation**: See `../docs/`
- **Scheduling Engine Integration**: See `../scheduling_system/README.md`
- **Data Model Reference**: See `../docs/DATA_MODEL.md`

---

**Development Status**: Nearly complete implementation  
**Validation Phase**: In progress  
**Last Updated**: October 2025
