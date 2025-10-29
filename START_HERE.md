# 🏰 EntroPit - START HERE

**Welcome to EntroPit!** You're about to build a probabilistic dungeon generator using thermodynamic computing principles.

---

## ⚡ Quick Start (5 Minutes)

### 1. Install Dependencies

```bash
pip install jax jaxlib thrml matplotlib pillow numpy
```

### 2. Run Your First Dungeon

```bash
python entropit_quickstart.py
```

### 3. View Results

Open `entropit_dungeons.png` to see your generated dungeons!

**🎉 That's it!** You just sampled from a probability distribution over dungeons using Gibbs sampling.

---

## 📚 Documentation Guide

### New to the Project?
**→ Read:** `README_ENTROPIT.md`
- What EntroPit does
- Why it matters
- Feature roadmap
- Connection to Extropic hardware

### Ready to Build?
**→ Read:** `GETTING_STARTED.md`
- Step-by-step tutorial
- Parameter tuning guide
- Common issues & solutions
- Next steps

### Want Deep Understanding?
**→ Read:** `ENTROPIT_ARCHITECTURE.md`
- Mathematical formulation
- PGM design
- Energy functions
- Sampling strategies
- Research directions

### Organizing Your Code?
**→ Read:** `ENTROPIT_PROJECT_STRUCTURE.md`
- Project layout
- Development phases
- File organization
- Recommended workflow

---

## 🎯 Your Learning Path

### Level 1: Explorer 🗺️
**Goal:** Understand what's happening

1. ✅ Run `entropit_quickstart.py`
2. Read `README_ENTROPIT.md` (15 min)
3. Tweak parameters in quickstart script
4. Read `GETTING_STARTED.md` (30 min)
5. Generate 20+ dungeons with different settings

**You'll learn:** How probabilistic generation differs from traditional algorithms

---

### Level 2: Builder 🔨
**Goal:** Create your own features

1. Read `ENTROPIT_ARCHITECTURE.md` sections 1-3 (45 min)
2. Add a fixed spawn point constraint
3. Implement connectivity checking (NetworkX)
4. Build a simple Gradio UI
5. Export dungeons to JSON

**You'll learn:** How to map design goals to energy functions

---

### Level 3: Innovator 🚀
**Goal:** Push the boundaries

1. Read full `ENTROPIT_ARCHITECTURE.md` (1-2 hours)
2. Implement categorical nodes (doors, treasure)
3. Add path-length constraints
4. Create custom energy terms
5. Build annealing schedules
6. Benchmark vs. traditional methods

**You'll learn:** Why probabilistic computing is powerful for constraint satisfaction

---

### Level 4: Researcher 🔬
**Goal:** Contribute new ideas

1. Study THRML documentation
2. Implement metroidvania key/lock constraints
3. Multi-floor 3D dungeons
4. Style transfer from existing dungeons
5. Real-time editing during gameplay
6. Write a paper/blog post!

**You'll learn:** The future of procedural content generation

---

## 🎨 Project Ideas by Difficulty

### Easy (Weekend Project)
- [ ] Parameter tuning dashboard (Gradio)
- [ ] ASCII art renderer
- [ ] Dungeon gallery generator (100s of dungeons)
- [ ] Export to Unity JSON format

### Medium (Week Project)
- [ ] Interactive constraint painting
- [ ] Connectivity guarantee
- [ ] Treasure/enemy placement PGM
- [ ] Annealing schedule visualizer
- [ ] Benchmark vs. cellular automata

### Hard (Month Project)
- [ ] Multi-floor dungeons with stairs
- [ ] Metroidvania progression (keys/doors)
- [ ] Learn style from existing dungeons (RBM)
- [ ] Real-time level editing in game engine
- [ ] 3D voxel dungeons

### Research (Open-Ended)
- [ ] Narrative constraint modeling
- [ ] Multi-agent design optimization
- [ ] Hardware simulation (Extropic chip model)
- [ ] Comparison with diffusion models
- [ ] Application to other domains (circuits, molecules, etc.)

---

## 🛠️ Tools You'll Use

### Core Stack
- **JAX**: GPU-accelerated array computing
- **THRML**: Block Gibbs sampling for PGMs
- **NumPy**: Array manipulation
- **Matplotlib**: Visualization

### Optional Additions
- **Gradio**: Web UI (recommended!)
- **NetworkX**: Graph algorithms (connectivity)
- **Pillow**: Image processing
- **SciPy**: Advanced spatial algorithms

---

## 🔥 Cool Demos to Build

### 1. Live Sampling Visualizer
Show the dungeon evolving in real-time as Gibbs sampling runs. Watch order emerge from chaos!

### 2. Constraint Painting
Let users paint "must be wall" / "must be floor" regions, then sample dungeons satisfying those constraints.

### 3. Style Mixer
Generate dungeons that blend two styles (e.g., 60% cave-like + 40% fortress-like).

### 4. Infinite World Generator
Generate only the parts of the world the player has discovered, maintaining global coherence.

### 5. Difficulty Tuner
Slider from "easy" to "hard" that adjusts enemy density, path complexity, etc.

---

## 🚨 Common Questions

### Q: I don't understand Gibbs sampling. Can I still use this?

**A:** Yes! Start with the quickstart. You can build cool stuff before understanding the math. The architecture doc will make more sense after you've played with it.

### Q: Do I need a GPU?

**A:** No, but it helps. JAX works fine on CPU for small dungeons (≤20×20). GPU shines for larger grids or batch generation.

### Q: How is this different from WaveFunctionCollapse?

**A:** WFC uses local constraints only. EntroPit can handle global constraints (path length, treasure distribution) and learns from energy functions rather than tile rules.

### Q: When will Extropic hardware be available?

**A:** Unknown publicly. EntroPit works today on GPUs and will be *much* faster when hardware ships.

### Q: Can I use this in my game?

**A:** Yes! MIT license (check THRML's license too). Ship it!

---

## 🎓 Learning Resources

### Probability & Sampling
- [Markov Chain Monte Carlo (MCMC) Introduction](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo)
- [Gibbs Sampling Explained](https://en.wikipedia.org/wiki/Gibbs_sampling)

### Ising Model
- [Ising Model Basics](https://en.wikipedia.org/wiki/Ising_model)
- [Statistical Physics for ML](https://arxiv.org/abs/1803.08823)

### JAX
- [JAX Quickstart](https://jax.readthedocs.io/en/latest/quickstart.html)
- [JAX for Scientists](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)

### THRML
- [THRML Documentation](https://docs.thrml.ai/)
- [THRML GitHub](https://github.com/extropic-ai/thrml)

### Procedural Generation
- [Procedural Content Generation Wiki](http://pcg.wikidot.com/)
- [WaveFunctionCollapse](https://github.com/mxgmn/WaveFunctionCollapse)

---

## 💡 Tips for Success

### 1. Start Tiny
Begin with 8×8 grids and simple energy functions. Scale up once it works.

### 2. Visualize Everything
Print energies, plot convergence, animate sampling. Seeing is understanding.

### 3. Compare to Baselines
Generate dungeons with simple algorithms (random, cellular automata) to see the difference.

### 4. Embrace Randomness
Each run is different. That's a feature, not a bug!

### 5. Ask "Why?"
When something doesn't work, ask: "What energy landscape did I create?" Debug the math, not just the code.

---

## 🌟 Share Your Work!

Built something cool? Share it:
- Tag @extropic_ai on Twitter/X
- Post in THRML discussions
- Write a blog post
- Submit a PR with improvements

The thermodynamic computing community is small but growing. Your work matters!

---

## 🗺️ Roadmap Suggestion

### Week 1
- ✅ Run quickstart
- ✅ Read docs
- ✅ Experiment with parameters
- ✅ Add spawn constraint

### Week 2
- Build Gradio UI
- Add connectivity check
- Implement JSON export
- Create dungeon gallery

### Week 3
- Interactive constraint painting
- Annealing schedule tuning
- Benchmark performance
- Write tutorial blog post

### Week 4+
- Advanced features (your choice!)
- Integrate with game engine
- Research extensions
- Share with community

---

## 🚀 Ready to Begin?

Pick one:

**🏃 I want to code RIGHT NOW**
→ `python entropit_quickstart.py`

**📖 I want to understand first**
→ Read `README_ENTROPIT.md`

**🔧 I want to build something specific**
→ Read `GETTING_STARTED.md` then `ENTROPIT_ARCHITECTURE.md`

**🤔 I'm still not sure what this is**
→ Read the "What It Does" section in `README_ENTROPIT.md`

---

## 🎬 Let's Go!

```bash
# Create project directory
mkdir entropit-project && cd entropit-project

# Install dependencies
pip install jax jaxlib thrml matplotlib pillow numpy

# Copy quickstart script here (or write it from scratch!)

# Generate your first probabilistic dungeon
python entropit_quickstart.py

# View the result
# (opens entropit_dungeons.png)
```

**Welcome to the future of procedural generation.** 🔥

*Where physics meets game design.* ⚔️🏰

---

## 📧 Questions?

- Read the docs (they're comprehensive!)
- Check THRML GitHub issues
- Experiment and iterate
- Share your findings

**Most importantly: Have fun!** This is cutting-edge stuff made accessible.

---

*EntroPit - Let entropy design your dungeons.* 🌀

