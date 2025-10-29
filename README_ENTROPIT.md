# 🏰 EntroPit

**Probabilistic Dungeon Generation Powered by Thermodynamic Computing**

EntroPit is an interactive procedural dungeon generator that uses probabilistic graphical models (PGMs) and Gibbs sampling to create dungeons with user-defined constraints. Built on [THRML](https://github.com/extropic-ai/thrml), it simulates what's possible when deploying to Extropic's specialized thermodynamic computing hardware—where sampling complex probability distributions becomes dramatically faster and more energy-efficient.

<div align="center">
  
  **🎲 Constraint-Based • ⚡ GPU-Accelerated • 🔥 Thermodynamically Inspired**
  
</div>

---

## 🎯 What It Does

EntroPit generates dungeons by modeling each tile as a node in an Ising-like energy-based model. Instead of using traditional procedural algorithms, it:

1. **Defines an energy landscape** where "good" dungeons have low energy
2. **Sets constraints** (connectivity, room sizes, treasure placement, etc.)
3. **Samples from the Boltzmann distribution** using block Gibbs sampling
4. **Converges to valid dungeons** that satisfy your creative vision

Think of it as **painting with probability** rather than deterministic algorithms.

---

## 🔬 Why This Matters

Traditional dungeon generators use hardcoded rules (BSP trees, cellular automata, etc.). EntroPit demonstrates a fundamentally different approach:

- **Declarative Design**: Define *what* you want (constraints), not *how* to build it
- **Soft Constraints**: Balance competing goals (reachability vs. openness vs. interestingness)
- **Hardware-Ready**: When Extropic's chips arrive, this scales to massive dungeons in real-time
- **Probabilistic by Nature**: Each generation explores a different path through possibility space

This is a glimpse of how future game engines might leverage probabilistic accelerators for:
- Real-time procedural content generation
- Constraint satisfaction (NPC behavior, quest generation)
- Combinatorial optimization (loot distribution, difficulty balancing)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- JAX (CPU or GPU)
- THRML

### Installation

```bash
# Clone this repo (or create a new project directory)
mkdir entropit
cd entropit

# Install dependencies
pip install jax jaxlib  # or jax[cuda12] for GPU
pip install thrml
pip install gradio matplotlib numpy pillow

# If you cloned the thrml repo and want to use local version:
# cd /path/to/thrml
# pip install -e .
```

### Run Your First Dungeon

```bash
# Coming soon - we'll build this together!
python entropit_app.py
```

Then open `http://localhost:7860` in your browser and start generating!

---

## 🎨 Features Roadmap

### Phase 1: Core Engine ✨ (We Are Here)
- [x] Project setup
- [ ] Basic Ising model formulation for 2D grid
- [ ] Connectivity constraints (ensure all rooms reachable)
- [ ] Wall/floor/corridor energy terms
- [ ] Simple Gradio UI with generate button
- [ ] Visualization of sampling convergence

### Phase 2: Interactive Constraints
- [ ] User can paint "must be wall/floor" constraints
- [ ] Room size distribution control
- [ ] Path length constraints (spawn to exit)
- [ ] Multiple sampling schedules (fast vs. high quality)

### Phase 3: Dungeon Furniture
- [ ] Treasure/loot placement
- [ ] Enemy spawn points with difficulty curves
- [ ] Door and key constraints (metroidvania-style)
- [ ] Special room types (boss room, shop, puzzle)

### Phase 4: Advanced PGM Features
- [ ] Heterogeneous nodes (not just binary wall/floor)
- [ ] Multi-floor dungeons with staircase constraints
- [ ] Style transfer (learn from existing dungeons)
- [ ] Export to game engines (Godot, Unity formats)

### Phase 5: Performance & Hardware Sim
- [ ] Benchmark classical vs. THRML sampling
- [ ] "Extropic accelerator mode" (simulate hardware speedup)
- [ ] Batch generation (1000s of dungeons)
- [ ] Real-time editing with live re-sampling

---

## 🧮 How It Works (Technical Overview)

### The Energy Function

Each dungeon configuration \( s \) has an energy:

$$E(s) = E_{\text{wall}} + E_{\text{conn}} + E_{\text{room}} + E_{\text{constraint}}$$

Where:
- **\( E_{\text{wall}} \)**: Penalizes isolated walls (encourages corridors)
- **\( E_{\text{conn}} \)**: Rewards connectivity between regions
- **\( E_{\text{room}} \)**: Encourages room-like open spaces
- **\( E_{\text{constraint}} \)**: Hard constraints from user input

### Sampling Process

1. Model each tile as a `SpinNode` (wall = -1, floor = +1)
2. Define pairwise interactions based on adjacency
3. Use **block Gibbs sampling** (color graph, update blocks in parallel)
4. Anneal temperature from high → low for better convergence
5. Render final configuration

### Why THRML?

THRML handles the complex orchestration of:
- Block coloring and parallel updates
- Interaction management between nodes
- Efficient state representation for JAX
- Integration with energy-based models

On Extropic hardware, the core sampling loop would run orders of magnitude faster, enabling:
- Real-time generation during gameplay
- Massive parallel generation (world-building)
- More complex constraints without performance penalty

---

## 🎮 Use Cases

- **Game Developers**: Drop-in dungeon generator with art-directable constraints
- **Roguelike Designers**: Generate infinite varied dungeons with guaranteed properties
- **Educators**: Teach probabilistic inference and PGMs through visual examples
- **Researchers**: Benchmark sampling algorithms on structured constraint problems
- **Extropic Evangelists**: Demonstrate thermodynamic computing applications

---

## 🤝 Contributing

This is a demo/educational project, but contributions are welcome!

Ideas:
- New constraint types
- Better energy functions
- UI improvements
- 3D dungeon support
- Export formats for game engines

---

## 📚 Learning Resources

- **THRML Docs**: https://docs.thrml.ai/
- **Ising Models**: Start with spin glasses and constraint satisfaction
- **Procedural Generation**: Look into WaveFunctionCollapse for comparison
- **Gibbs Sampling**: Understand MCMC and how thermodynamic systems equilibrate

---

## 🔮 The Extropic Vision

Extropic is building specialized hardware that makes sampling from certain probability distributions vastly more efficient by leveraging physical thermodynamics. EntroPit is a taste of what becomes possible:

> *"What if procedural generation happened at the speed of physics?"*

When the hardware ships, this project could generate photorealistic 3D dungeons in milliseconds, enable real-time constraint editing during gameplay, or power entire procedural worlds.

---

## 📄 License

This project is MIT licensed (same as THRML). Built as an educational demonstration of thermodynamic computing applications.

---

## 🙏 Acknowledgments

- **Extropic AI** for building THRML and pushing probabilistic computing forward
- The JAX team for making this kind of GPU-accelerated sampling accessible
- Classic roguelikes (NetHack, Dungeon Crawl) for inspiration

---

**Let's build dungeons with physics.** 🔥⚔️

---

## Next Steps

Ready to start building? Check out:
1. `docs/architecture.md` - Detailed PGM formulation
2. `entropit_app.py` - Main application (we'll build this next!)
3. `examples/simple_dungeon.py` - Minimal working example

*EntroPit - Where entropy meets the dungeon pit.* 🌀

