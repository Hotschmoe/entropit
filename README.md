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
pip install -r requirements.txt
```

For GPU support:
```bash
pip install jax[cuda12]  # Use cuda11 for older CUDA versions
```

### Option 1: Interactive UI (Recommended)

```bash
python entropit_ui.py
```

Then open `http://localhost:7860` in your browser to:
- Generate dungeons with different algorithms
- Compare THRML vs traditional methods
- Adjust parameters interactively (now fully working!)
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

### Option 3: Command Line Demo

```bash
python entropit_quickstart.py  # Simple demo
python entropit.py             # Full comparison demo
python benchmark.py            # Performance comparison
```

---

## How to Experiment

### Tweak Parameters (Now Interactive!)

Use the Gradio UI (`python entropit_ui.py`) to adjust:
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

See `ARCHITECTURE.md` for advanced techniques like clamping specific tiles.

---

## Documentation & Files

| File | Purpose |
|------|---------|
| **`README.md`** | This file - project overview and getting started |
| **`ARCHITECTURE.md`** | Mathematical formulation, energy functions, PGM design |
| **`entropit.py`** | **Main module** - `generate_thrml()` and `generate_traditional()` APIs |
| **`entropit_ui.py`** | Interactive Gradio web interface |
| **`entropit_quickstart.py`** | Simple command-line demo |
| **`traditional_generators.py`** | Baseline algorithms (Random, CA, BSP, Drunkard's Walk) |
| **`benchmark.py`** | Comprehensive comparison framework |
| **`requirements.txt`** | All dependencies |

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

- [THRML Documentation](https://docs.thrml.ai/)
- [THRML GitHub](https://github.com/extropic-ai/thrml)
- [Ising Model (Wikipedia)](https://en.wikipedia.org/wiki/Ising_model)
- [Gibbs Sampling (Wikipedia)](https://en.wikipedia.org/wiki/Gibbs_sampling)

---

## License

MIT License. Built as an educational demonstration of thermodynamic computing applications.

---

**Let's build dungeons with physics.**

