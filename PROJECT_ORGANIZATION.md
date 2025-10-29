# EntroPit - Project Organization Guide

## Current Structure (Phase 1)

```
entropit/
├── entropit.py                 # Main API - generate_thrml(), generate_traditional()
├── entropit_ui.py              # Gradio web interface
├── entropit_quickstart.py      # Simple command-line demo
├── traditional_generators.py   # Classical algorithms (Random, CA, BSP, Drunkard)
├── benchmark.py                # Performance comparison framework
│
├── output/                     # Generated files (gitignored)
│   ├── *.png                   # Visualizations
│   └── *.json                  # Exported dungeons (future)
│
├── README.md                   # Project overview
├── ARCHITECTURE.md             # Technical deep-dive
├── TODO.md                     # Development roadmap
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git exclusions
└── PROJECT_ORGANIZATION.md     # This file
```

### File Responsibilities

**`entropit.py`** - Main module
- `generate_thrml()` - THRML-based generation with full parameter control
- `generate_traditional()` - Traditional algorithm wrapper
- `analyze_dungeon()` - Quality metrics

**`entropit_ui.py`** - User interface
- Gradio web app
- Parameter controls for all generation methods
- Side-by-side comparison
- Real-time visualization

**`entropit_quickstart.py`** - Getting started
- Standalone demo (no imports from other modules)
- Good for learning
- Copy-paste friendly

**`traditional_generators.py`** - Baselines
- Random, Cellular Automata, BSP, Drunkard's Walk
- Pure NumPy implementations
- Used for benchmarking

**`benchmark.py`** - Analysis
- Performance metrics
- Connectivity checking
- Playability scoring
- Comparison framework

## When to Reorganize

### Keep Current Structure If:
- ✅ Under 10 Python files
- ✅ Single developer or small team
- ✅ Focused on experimentation/research
- ✅ Fast iteration is priority

### Move to Organized Structure When:
- ❌ 10+ Python files
- ❌ Adding comprehensive test suite
- ❌ Preparing for PyPI package
- ❌ Multiple contributors
- ❌ Building production features

## Future Structure (Phase 2)

When the project grows, reorganize to:

```
entropit/
├── src/
│   └── entropit/
│       ├── __init__.py          # Public API exports
│       ├── core/
│       │   ├── __init__.py
│       │   ├── thrml_generator.py
│       │   ├── traditional.py
│       │   └── constraints.py   # Constraint system
│       ├── analysis/
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   └── benchmark.py
│       ├── ui/
│       │   ├── __init__.py
│       │   └── gradio_app.py
│       └── export/
│           ├── __init__.py
│           ├── json_format.py
│           └── tilemap.py
│
├── examples/
│   ├── quickstart.py
│   ├── demo.py
│   ├── advanced_constraints.py
│   └── notebooks/
│       ├── tutorial.ipynb
│       └── benchmark_analysis.ipynb
│
├── tests/
│   ├── __init__.py
│   ├── test_generation.py
│   ├── test_constraints.py
│   ├── test_analysis.py
│   └── conftest.py
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── API.md
│   ├── CONTRIBUTING.md
│   └── tutorials/
│       ├── getting_started.md
│       └── advanced_usage.md
│
├── benchmarks/                  # Separate from tests
│   ├── performance.py
│   └── quality_comparison.py
│
├── output/                      # Generated files
│
├── README.md
├── TODO.md
├── requirements.txt
├── requirements-dev.txt         # Dev dependencies (pytest, etc.)
├── setup.py                     # Package configuration
├── pyproject.toml               # Modern Python packaging
├── .gitignore
└── LICENSE
```

## Migration Path

### Step 1: Add Tests (Do This Soon)
```bash
mkdir tests
# Create basic tests for generate_thrml and generate_traditional
```

### Step 2: Move Examples (When You Have 3+ Example Scripts)
```bash
mkdir examples
mv entropit_quickstart.py examples/quickstart.py
mv entropit.py examples/demo.py  # Main demo script
```

### Step 3: Create Package (When Ready for Distribution)
```bash
mkdir -p src/entropit
# Move core files into src/entropit/
# Create __init__.py with public API
# Add setup.py for pip installation
```

### Step 4: Separate Benchmarks (When You Have Multiple Benchmark Scripts)
```bash
mkdir benchmarks
mv benchmark.py benchmarks/
```

## Best Practices

### File Naming
- **Modules**: `lowercase_with_underscores.py`
- **Classes**: `CapitalizedWords`
- **Functions**: `lowercase_with_underscores()`
- **Constants**: `UPPER_CASE_WITH_UNDERSCORES`

### Imports
```python
# Current (flat structure)
from entropit import generate_thrml, generate_traditional

# Future (package structure)
from entropit import generate_thrml, generate_traditional
from entropit.analysis import benchmark_generator
from entropit.constraints import ConstraintPainter
```

### Version Control
```bash
# Commit current state before big refactoring
git commit -am "Snapshot before reorganization"

# Create refactor branch
git checkout -b refactor-project-structure

# Test thoroughly before merging
pytest tests/
python examples/quickstart.py
```

## Decision Checklist

Before reorganizing, ask:

- [ ] Do we have tests to ensure nothing breaks?
- [ ] Is the current structure actually causing problems?
- [ ] Will the new structure make development faster?
- [ ] Is this the right time (vs. adding features)?

**Generally**: Don't reorganize until pain > reorganization_cost

## Recommended Timeline

**Now (Phase 1)**: Current flat structure  
**After 5 features**: Add `tests/` directory  
**After 10 files**: Move to `examples/` pattern  
**Before PyPI**: Full package structure with `src/`

---

**Current Status**: Phase 1 (Flat structure)  
**Next Milestone**: Add tests, then keep flat until ~10 Python files  
**Future**: Full package when preparing for distribution

