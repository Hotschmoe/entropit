# 🏰 EntroPit

**Probabilistic Dungeon Generation Powered by Thermodynamic Computing**

EntroPit is a procedural dungeon generator that uses probabilistic graphical models (PGMs) and Gibbs sampling to create dungeons. Built on [THRML](https://github.com/extropic-ai/thrml), it demonstrates what's possible with thermodynamic computing—where sampling complex probability distributions becomes dramatically faster and more energy-efficient.

<div align="center">
  
  **🎲 Constraint-Based • ⚡ GPU-Accelerated • 🔥 Thermodynamically Inspired**
  
</div>

---

## 🎯 What It Does

EntroPit generates dungeons by modeling each tile as a node in an Ising-like energy-based model:

1. **Defines an energy landscape** where "good" dungeons have low energy
2. **Samples from the Boltzmann distribution** using block Gibbs sampling
3. **Converges to valid dungeons** that emerge from the probability distribution

Think of it as **painting with probability** rather than deterministic algorithms.

---

## 🚀 Quick Start

### Installation

```bash
pip install jax jaxlib thrml matplotlib pillow numpy
```

For GPU support:
```bash
pip install jax[cuda12]  # Use cuda11 for older CUDA versions
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### Run Your First Dungeon

```bash
python entropit_quickstart.py
```

This generates a 12×12 dungeon grid and saves visualizations to `entropit_dungeons.png`.

**You just sampled from a probability distribution over dungeons!** 🎉

---

## 🧪 How to Experiment

### Tweak Parameters

Open `entropit_quickstart.py` and modify:

**Grid Size:**
```python
dungeons, nodes = create_simple_dungeon(grid_size=20, seed=42)
```

**Edge Bias** (more negative = thicker walls):
```python
biases.append(-2.0 if is_edge else 0.5)
```

**Coupling Strength** (higher = bigger rooms):
```python
weights = jnp.ones(len(edges)) * 0.8
```

**Temperature** (higher = more structured):
```python
beta = jnp.array(2.0)
```

### Add Constraints

Force a spawn point in the center by clamping specific nodes—see `ARCHITECTURE.md` for advanced techniques.

---

## 📚 Documentation

- **`ARCHITECTURE.md`** - Mathematical formulation, energy functions, PGM design, and research directions
- **`entropit_quickstart.py`** - Annotated code showing the complete pipeline
- **`requirements.txt`** - All dependencies

---

## 🎯 Why This Matters

Traditional dungeon generators use hardcoded rules (BSP trees, cellular automata). EntroPit demonstrates a fundamentally different approach:

- **Declarative Design**: Define *what* you want (constraints), not *how* to build it
- **Soft Constraints**: Balance competing goals naturally through energy minimization
- **Hardware-Ready**: Designed for Extropic's thermodynamic computing chips
- **Probabilistic**: Each generation explores a different path through possibility space

This approach applies beyond dungeons:
- Real-time procedural content generation
- Constraint satisfaction (NPC behavior, quest generation)
- Combinatorial optimization (loot distribution, difficulty balancing)

---

## 🎨 Future Features

- Interactive constraint painting (Gradio UI)
- Connectivity guarantees (flood-fill + resampling)
- Categorical nodes (doors, treasure, enemies)
- Multi-floor dungeons
- Export to game engines (JSON, Godot, Unity)
- Metroidvania key/lock constraints
- Style transfer from existing dungeons

---

## 🔮 The Extropic Vision

Extropic is building specialized hardware that makes sampling from probability distributions vastly more efficient by leveraging physical thermodynamics.

> *"What if procedural generation happened at the speed of physics?"*

When the hardware ships, this approach could generate massive dungeons in milliseconds or enable real-time constraint editing during gameplay.

---

## 📚 Learning Resources

- [THRML Documentation](https://docs.thrml.ai/)
- [THRML GitHub](https://github.com/extropic-ai/thrml)
- [Ising Model (Wikipedia)](https://en.wikipedia.org/wiki/Ising_model)
- [Gibbs Sampling (Wikipedia)](https://en.wikipedia.org/wiki/Gibbs_sampling)

---

## 📄 License

MIT License. Built as an educational demonstration of thermodynamic computing applications.

---

**Let's build dungeons with physics.** 🔥⚔️

