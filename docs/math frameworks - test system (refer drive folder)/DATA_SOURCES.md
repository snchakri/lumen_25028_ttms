# Data Source Documentation

## Overview

The test data generator supports **extensible JSON data sources** for realistic entity generation. Instead of hardcoding values, generators can load data from JSON files, making it easy to customize data for different regions, domains, or scenarios.

## Directory Structure

```
data/
├── institutions/
│   ├── institution_names.json      # Institution names with metadata
│   └── institution_types.json      # Types of institutions
├── departments/
│   └── department_names.json       # Department names with codes
├── programs/
│   └── degree_types.json           # Degree types with credit requirements
├── courses/
│   ├── course_titles.json          # Course titles with metadata
│   ├── course_descriptions.json    # Course descriptions
│   └── course_prefixes.json        # Department-specific prefixes
├── people/
│   ├── first_names.json            # First names
│   ├── last_names.json             # Last names
│   └── faculty_titles.json         # Academic titles
├── facilities/
│   ├── room_types.json             # Room types with capacity ranges
│   ├── building_names.json         # Building names and codes
│   └── equipment_types.json        # Equipment types
└── temporal/
    ├── shift_patterns.json         # Shift timing patterns
    └── semester_patterns.json      # Semester configurations
```

## JSON File Format

### Simple Lists

For simple string lists (e.g., first names):

```json
[
  "Rajesh",
  "Priya",
  "Amit",
  "Sneha"
]
```

### Objects with Metadata

For complex entities with attributes:

```json
[
  {
    "name": "National Institute of Technology",
    "type": "PUBLIC",
    "state": "Karnataka"
  },
  {
    "name": "Indian Institute of Technology",
    "type": "AUTONOMOUS",
    "state": "Maharashtra"
  }
]
```

### Objects with Constraints

For entities with validation rules or constraints:

```json
[
  {
    "type": "Lecture Hall",
    "capacity_min": 50,
    "capacity_max": 200
  },
  {
    "type": "Classroom",
    "capacity_min": 30,
    "capacity_max": 60
  }
]
```

## Using the Data Loader

### In Generators

```python
from src.data_loaders import JSONDataLoader

class MyGenerator(BaseGenerator):
    def __init__(self, config, state_manager=None):
        super().__init__(config, state_manager)
        self.data_loader = JSONDataLoader()
        self._names = []
    
    def load_source_data(self) -> bool:
        # Load data from JSON file
        self._names = self.data_loader.load_file(
            "institutions/institution_names.json"
        )
        
        # Fallback to synthetic if file not found
        if not self._names:
            logger.warning("Using synthetic data fallback")
            self._names = [
                {"name": f"Institution {i}"} 
                for i in range(100)
            ]
        
        return True
    
    def generate_entities(self) -> List[Dict[str, Any]]:
        # Get random item from loaded data
        institution_data = self.data_loader.get_random_item(
            "institutions/institution_names.json"
        )
        
        # Get multiple items
        names = self.data_loader.get_random_items(
            "people/first_names.json",
            count=10,
            unique=True
        )
        
        # Use filtered selection
        stem_depts = self.data_loader.get_random_items(
            "departments/department_names.json",
            count=5,
            filter_fn=lambda d: d.get("domain") == "Engineering"
        )
```

### Standalone Usage

```python
from src.data_loaders import load_json_data, get_random_item

# Load entire file
institutions = load_json_data("institutions/institution_names.json")

# Get random item
random_dept = get_random_item("departments/department_names.json")

# Get random item with filter
cs_dept = get_random_item(
    "departments/department_names.json",
    filter_fn=lambda d: d["code"] == "CS"
)
```

## Adding New Data Sources

### Step 1: Create JSON File

Create a new JSON file in the appropriate subdirectory:

```bash
# Example: Add new course descriptions
data/courses/advanced_topics.json
```

### Step 2: Define Data Structure

```json
[
  {
    "title": "Quantum Computing",
    "level": 400,
    "credits": 3,
    "prerequisites": ["CS301", "MATH240"]
  },
  {
    "title": "Blockchain Technology",
    "level": 400,
    "credits": 3,
    "prerequisites": ["CS302"]
  }
]
```

### Step 3: Use in Generator

```python
def load_source_data(self) -> bool:
    self.advanced_courses = self.data_loader.load_file(
        "courses/advanced_topics.json"
    )
    return True
```

## Data File Guidelines

### 1. **Use Arrays of Objects**
- Always use arrays as the root element
- Use objects for structured data with multiple fields
- Use simple strings/numbers for single-value lists

### 2. **Include Metadata**
- Add relevant metadata fields for filtering
- Include constraints (min/max values, ranges)
- Add categorization fields (type, domain, level)

### 3. **Keep Files Focused**
- One file per data type or category
- Split large datasets into multiple files
- Use descriptive filenames

### 4. **Follow Naming Conventions**
- Use snake_case for filenames
- Use lowercase with underscores
- Use plural names for lists (e.g., `first_names.json`)

### 5. **Validate JSON**
- Ensure valid JSON syntax
- Test files before deployment
- Use JSON validators/linters

## Example Data Files

### institution_names.json
```json
[
  {
    "name": "National Institute of Technology",
    "type": "PUBLIC",
    "state": "Karnataka"
  }
]
```

### department_names.json
```json
[
  {
    "name": "Computer Science",
    "code": "CS",
    "domain": "Engineering"
  }
]
```

### room_types.json
```json
[
  {
    "type": "Lecture Hall",
    "capacity_min": 50,
    "capacity_max": 200
  }
]
```

## Fallback Behavior

If a JSON file is missing or invalid:

1. **Logger Warning**: Warning logged with file path
2. **Synthetic Fallback**: Generator uses synthetic data generation
3. **Graceful Degradation**: Generation continues without interruption
4. **No Exceptions**: System never crashes due to missing data files

## Performance Considerations

### Caching

- Data files are loaded once and cached in memory
- Subsequent accesses use cached data (no file I/O)
- Cache can be cleared for testing: `loader.clear_cache()`

### Memory Usage

- Small files (<1MB): Load entirely into memory
- Large files: Consider splitting into smaller files
- Typical usage: 10-50 JSON files, 1-10KB each

## Best Practices

1. **Start Small**: Begin with essential data files
2. **Test Fallbacks**: Verify synthetic fallback works
3. **Version Data**: Keep data files in version control
4. **Document Schema**: Add comments in README about structure
5. **Validate Regularly**: Test data file integrity

## CLI Integration

Use `--data-dir` option to specify custom data directory:

```bash
python cli.py generate --data-dir /path/to/custom/data
```

This allows:
- Multiple data sets (dev, test, production)
- Region-specific data (US, India, Europe)
- Domain-specific data (engineering, healthcare, business)

## Troubleshooting

### File Not Found
```
WARNING: Data file not found: data/institutions/institution_names.json
```
**Solution**: Check file path and ensure file exists

### Invalid JSON
```
ERROR: Invalid JSON in data/courses/titles.json: Expecting ',' delimiter
```
**Solution**: Validate JSON syntax using online validator

### Empty Results
```
WARNING: Filter removed all items from departments/department_names.json
```
**Solution**: Check filter function logic or expand data set

## Summary

The extensible data source system provides:

✅ **Flexibility**: Easy to customize data for different scenarios  
✅ **Realism**: Use real institution/course names instead of synthetic  
✅ **Maintainability**: Update data without code changes  
✅ **Graceful Fallback**: Never blocks generation if files missing  
✅ **Performance**: Caching for fast repeated access
