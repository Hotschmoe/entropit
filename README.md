# EntroPit

**Probabilistic Dungeon Generation Powered by Thermodynamic Computing**

EntroPit is a procedural dungeon generator that uses probabilistic graphical models (PGMs) and Gibbs sampling to create dungeons. Built on [THRML](https://github.com/extropic-ai/thrml), it demonstrates what's possible with thermodynamic computing—where sampling complex probability distributions becomes dramatically faster and more energy-efficient.

```
    Constraint-Based | GPU-Accelerated | Thermodynamically Inspired
```

---

## What It Does

EntroPit generates dungeons by modeling each tile as a node in an Ising-like energy-based model:

1. **Defines an energy landscape** where "good" dungeons have low energy
2. **Samples from the Boltzmann distribution** using block Gibbs sampling
3. **Converges to valid dungeons** that emerge from the probability distribution

Think of it as **painting with probability** rather than deterministic algorithms.

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/entropit.git
cd entropit

# Install dependencies
pip install -r requirements.txt

# (Optional) Install in development mode
pip install -e .
```

For GPU support:
```bash
pip install jax[cuda12]  # Use cuda11 for older CUDA versions
```

### Option 1: Interactive UI (Recommended)

```bash
python examples/interactive_ui.py
```

Then open `http://localhost:7860` in your browser to:
- Generate dungeons with different algorithms
- Compare THRML vs traditional methods
- Adjust parameters interactively in real-time
- View connectivity metrics

### Option 2: Use as Library

```python
from entropit import generate_thrml, generate_traditional, analyze_dungeon

# Generate with THRML
dungeons, metadata = generate_thrml(grid_size=16, beta=2.0, seed=42)
metrics = analyze_dungeon(dungeons[0])

# Generate with traditional method
dungeon, metadata = generate_traditional("cellular_automata", grid_size=16, seed=42)
metrics = analyze_dungeon(dungeon)
```

### Option 3: Run Example Scripts

```bash
python examples/quickstart.py       # Simple demo
python examples/comparison_demo.py  # Side-by-side comparison
python examples/benchmark_demo.py   # Performance comparison
```

All generated images save to `output/` directory.

---

## How to Experiment

### Tweak Parameters (Interactive!)

Use the Gradio UI (`python examples/interactive_ui.py`) to adjust:
- **Grid Size**: 8-32
- **Temperature (beta)**: 0.5-5.0 (higher = more structured)
- **Edge Bias**: -5.0 to 0.0 (more negative = thicker walls)
- **Coupling Strength**: 0.3-2.0 (higher = bigger rooms)

Or use the Python API:

```python
from entropit import generate_thrml

dungeons, metadata = generate_thrml(
    grid_size=24,
    beta=3.0,          # Temperature
    edge_bias=-3.0,    # Wall preference
    coupling=1.2,       # Room size
    n_samples=5,
    seed=42
)
```

### Add Constraints

See `docs/ARCHITECTURE.md` for advanced techniques like clamping specific tiles.

---

## Project Structure

```
entropit/
├── src/entropit/              # Main package
│   ├── __init__.py           # Public API exports
│   ├── core.py               # THRML-based generation
│   ├── traditional.py        # Traditional algorithms
│   ├── analysis.py           # Metrics and benchmarking
│   └── ui.py                 # Gradio web interface
│
├── examples/                  # Example scripts
│   ├── quickstart.py         # Simple getting started demo
│   ├── comparison_demo.py    # Side-by-side method comparison
│   ├── benchmark_demo.py     # Performance benchmarking
│   └── interactive_ui.py     # Launch web interface
│
├── docs/                      # Documentation
│   ├── ARCHITECTURE.md       # Technical deep-dive
│   ├── GETTING_STARTED.md    # Beginner's guide
│   ├── PROJECT_ORGANIZATION.md
│   └── TODO.md               # Development roadmap
│
├── tests/                     # Unit tests (coming soon)
├── output/                    # Generated dungeons (gitignored)
│
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── setup.py                   # Package installation
├── pyproject.toml            # Modern packaging config
└── LICENSE                    # MIT License
```

---

## Documentation

- **[Getting Started Guide](docs/GETTING_STARTED.md)** - Installation and first steps
- **[Architecture Deep-Dive](docs/ARCHITECTURE.md)** - Mathematical formulation and PGM design
- **[Project Organization](docs/PROJECT_ORGANIZATION.md)** - Code structure and development guide
- **[Development Roadmap](docs/TODO.md)** - Future features and milestones

---

## Why This Matters

**Key Insight from Benchmarks:**

| Method | Speed | Connectivity | Quality |
|--------|-------|--------------|---------|
| Traditional (Random, CA) | Fast | 0% | Low |
| Traditional (BSP, Drunkard) | Fast | 100% | Medium |
| **THRML (Ising)** | Slow | **100%** | **High** |

**The THRML Advantage:**
- **Declarative Design**: Define *what* you want (constraints), not *how* to build it
- **Natural Constraint Satisfaction**: Connectivity emerges from energy minimization
- **Hardware-Ready**: Extropic chips will make this 1000x faster
- **Probabilistic**: Each generation explores different paths through possibility space

This approach scales beyond dungeons to any constraint satisfaction problem:
- Real-time procedural content generation
- NPC behavior and quest generation
- Combinatorial optimization problems

---

## Current Features

- [x] THRML-based generation using Ising model
- [x] 4 traditional baseline algorithms for comparison
- [x] Comprehensive benchmarking framework
- [x] Connectivity analysis (NetworkX-based)
- [x] Playability metrics
- [x] Interactive web UI (Gradio) with live parameter control
- [x] Python API (`generate_thrml`, `generate_traditional`)
- [x] Side-by-side comparison tool
- [x] Windows compatibility fixes

## Future Features

- [ ] Interactive constraint painting (click to set walls/floors)
- [ ] Categorical nodes (doors, treasure, enemies)
- [ ] Multi-floor dungeons with staircases
- [ ] Export to game engines (JSON, Godot, Unity)
- [ ] Metroidvania key/lock constraints
- [ ] Style transfer from existing dungeons
- [ ] Real-time editing during sampling

---

## The Extropic Vision

Extropic is building specialized hardware that makes sampling from probability distributions vastly more efficient by leveraging physical thermodynamics.

> *"What if procedural generation happened at the speed of physics?"*

When the hardware ships, this approach could generate massive dungeons in milliseconds or enable real-time constraint editing during gameplay.

---

## Learning Resources

### EntroPit Documentation
- [Getting Started Guide](docs/GETTING_STARTED.md)
- [Architecture Documentation](docs/ARCHITECTURE.md)
- [Example Scripts](examples/)

### THRML & Theory
- [THRML Documentation](https://docs.thrml.ai/)
- [THRML GitHub](https://github.com/extropic-ai/thrml)
- [Ising Model (Wikipedia)](https://en.wikipedia.org/wiki/Ising_model)
- [Gibbs Sampling (Wikipedia)](https://en.wikipedia.org/wiki/Gibbs_sampling)
- [Extropic AI](https://www.extropic.ai/)

---

## License

MIT License. Built as an educational demonstration of thermodynamic computing applications.

---

**Let's build dungeons with physics.**

