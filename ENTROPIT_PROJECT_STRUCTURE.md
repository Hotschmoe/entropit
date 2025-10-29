# 📁 EntroPit Project Structure

Here's how to organize your EntroPit project files.

---

## Minimal Setup (Start Here)

```
entropit-project/
├── README_ENTROPIT.md           # Project overview & vision
├── GETTING_STARTED.md           # Step-by-step tutorial
├── ENTROPIT_ARCHITECTURE.md     # Technical deep-dive
├── entropit_requirements.txt    # Python dependencies
└── entropit_quickstart.py       # Minimal working example (RUN THIS FIRST!)
```

**Action**: Create this directory and copy/create these 5 files to get started.

---

## Phase 1: Core Engine (Weeks 1-2)

```
entropit-project/
├── [files from above]
├── entropit/
│   ├── __init__.py
│   ├── core.py              # Core dungeon generation logic
│   ├── models.py            # PGM/Energy function definitions
│   ├── sampling.py          # Wrapper around THRML sampling
│   └── visualization.py     # Matplotlib/PIL rendering
├── tests/
│   ├── test_basic.py        # Basic smoke tests
│   └── test_connectivity.py # Verify dungeons are valid
└── examples/
    ├── basic_dungeon.py     # Simple example
    └── animated_sampling.py # Watch evolution
```

---

## Phase 2: Interactive UI (Week 3)

```
entropit-project/
├── [previous files]
├── entropit/
│   ├── [previous modules]
│   ├── ui.py                # Gradio interface logic
│   └── constraints.py       # User-defined constraint handling
├── entropit_app.py          # Main Gradio app entry point
├── assets/
│   ├── logo.png
│   └── example_dungeons/    # Pre-generated examples
└── docs/
    └── user_guide.md        # End-user documentation
```

---

## Phase 3: Advanced Features (Week 4+)

```
entropit-project/
├── [previous files]
├── entropit/
│   ├── [previous modules]
│   ├── connectivity.py      # Graph algorithms for reachability
│   ├── furniture.py         # Treasure/enemy/door placement
│   ├── multi_floor.py       # 3D dungeon support
│   ├── export.py            # Game engine export formats
│   └── annealing.py         # Advanced sampling schedules
├── experiments/
│   ├── benchmark.py         # Performance testing
│   └── ablation_studies.py  # Parameter sensitivity
└── web/
    ├── index.html           # Optional custom web UI
    └── app.js
```

---

## Full Production Structure (Optional)

```
entropit-project/
├── README.md
├── LICENSE
├── pyproject.toml           # Package configuration
├── setup.py                 # Installation script
├── entropit/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── pgm.py           # PGM formulation
│   │   ├── energy.py        # Energy functions
│   │   └── sampler.py       # Sampling orchestration
│   ├── constraints/
│   │   ├── __init__.py
│   │   ├── connectivity.py
│   │   ├── spatial.py       # Room size, path length, etc.
│   │   └── user_defined.py
│   ├── furniture/
│   │   ├── __init__.py
│   │   ├── treasure.py
│   │   ├── enemies.py
│   │   └── doors.py
│   ├── rendering/
│   │   ├── __init__.py
│   │   ├── matplotlib_renderer.py
│   │   ├── ascii_renderer.py
│   │   └── tileset_renderer.py
│   ├── export/
│   │   ├── __init__.py
│   │   ├── json_format.py
│   │   ├── godot_format.py
│   │   └── unity_format.py
│   └── ui/
│       ├── __init__.py
│       ├── gradio_app.py
│       └── components.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── benchmarks/
├── examples/
│   ├── gallery/             # Collection of example dungeons
│   ├── tutorials/           # Jupyter notebooks
│   └── scripts/             # Standalone examples
├── docs/
│   ├── architecture.md
│   ├── api_reference.md
│   ├── tutorials/
│   └── research_notes.md
├── assets/
│   ├── tilesets/
│   ├── sprites/
│   └── fonts/
└── scripts/
    ├── generate_gallery.py
    ├── benchmark.py
    └── train_style_transfer.py
```

---

## File Descriptions

### Core Modules

**`core.py`**
- Main dungeon generation logic
- Wraps THRML API for ease of use
- Handles initialization, sampling, post-processing

**`models.py`**
- Energy function definitions
- Custom PGM configurations
- Preset dungeon styles (cave, fortress, maze, etc.)

**`sampling.py`**
- THRML sampling orchestration
- Annealing schedules
- Convergence monitoring

**`visualization.py`**
- Render dungeons to images
- ASCII art output
- Animation of sampling process

### Constraint System

**`constraints.py`**
- User-defined constraint handling
- Converts UI input to clamped blocks
- Validates constraint feasibility

**`connectivity.py`**
- Flood fill algorithms
- Connected component analysis
- Path finding (A*, Dijkstra)

### Furniture System

**`furniture.py`**
- Placement of game elements
- Treasure distribution
- Enemy spawn points
- Door/key logic for metroidvania style

### Export System

**`export.py`**
- JSON format (universal)
- Godot TileMap format
- Unity Prefab data
- Custom formats as needed

---

## Recommended Development Order

### Day 1: Setup
```bash
mkdir entropit-project && cd entropit-project
# Create the 5 minimal files
python entropit_quickstart.py  # Verify it works!
```

### Week 1: Core
```bash
mkdir entropit
# Refactor quickstart.py into entropit/core.py
# Add tests
```

### Week 2: Experiments
```bash
# Create parameter sweep scripts
# Test different energy functions
# Benchmark performance
```

### Week 3: UI
```bash
# Build Gradio interface
# Add constraint painting
# Polish visualization
```

### Week 4: Features
```bash
# Add connectivity checking
# Implement furniture placement
# Export to game engine format
```

---

## Configuration Files

### `pyproject.toml` (if packaging)

```toml
[project]
name = "entropit"
version = "0.1.0"
description = "Probabilistic dungeon generation with THRML"
authors = [{name = "Your Name"}]
requires-python = ">=3.10"
dependencies = [
    "jax>=0.4.0",
    "thrml>=0.1.3",
    "matplotlib>=3.7.0",
    "pillow>=10.0.0",
    "numpy>=1.24.0",
]

[project.optional-dependencies]
ui = ["gradio>=4.0.0"]
analysis = ["networkx>=3.0", "scipy>=1.10.0"]
dev = ["pytest>=7.0", "black>=25.0", "ruff>=0.11"]

[project.scripts]
entropit = "entropit.ui.gradio_app:main"
```

### `.gitignore`

```
# Python
__pycache__/
*.py[cod]
*.so
.Python
env/
venv/
*.egg-info/
dist/
build/

# JAX
.jax_cache/

# Generated files
entropit_dungeons.png
*.json
generated_dungeons/

# IDE
.vscode/
.idea/
*.swp
```

---

## Quick Reference: What Goes Where?

| I want to... | Edit this file |
|--------------|----------------|
| Change energy function | `models.py` or `core.py` |
| Add a constraint type | `constraints.py` |
| Modify sampling schedule | `sampling.py` |
| Change visualization style | `visualization.py` |
| Add UI element | `ui.py` or `entropit_app.py` |
| Add new export format | `export.py` |
| Test something | `tests/` or `examples/` |

---

## Start Simple!

Don't build the full structure right away. Begin with:

1. ✅ **Run `entropit_quickstart.py`**
2. Experiment with parameters
3. Add one feature at a time
4. Refactor when things get messy

**Premature organization is the root of all procrastination.** 😄

Start coding, organize later!

---

## Need Help Organizing?

If your project is growing and you need to refactor:

1. Keep `entropit_quickstart.py` as reference
2. Extract reusable functions to `entropit/core.py`
3. Move UI code to `entropit_app.py`
4. Add tests as you go

Most importantly: **Make it work, then make it clean.**

---

*Happy dungeon hacking!* ⚔️

