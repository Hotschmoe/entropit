# EntroPit Architecture

## Overview

EntroPit models dungeon generation as a **constraint satisfaction problem** solved through **probabilistic sampling** from an energy-based model (EBM). This document explains the mathematical formulation and implementation strategy.

---

## The Probabilistic Model

### Dungeon as a Probability Distribution

Rather than generating dungeons algorithmically, we define a probability distribution over all possible dungeon configurations:

$$P(s) = \frac{1}{Z} e^{-\beta E(s)}$$

Where:
- \( s \) is a dungeon configuration (state of all tiles)
- \( E(s) \) is the "energy" of that configuration
- \( \beta \) is inverse temperature (annealing parameter)
- \( Z \) is the partition function (normalization constant)

**Key insight**: Low-energy configurations are high-probability. By designing \( E(s) \) to make "good dungeons" have low energy, sampling from \( P(s) \) gives us good dungeons.

---

## Energy Function Design

The total energy is a sum of terms encoding different design goals:

$$E(s) = E_{\text{structure}} + E_{\text{connectivity}} + E_{\text{aesthetics}} + E_{\text{constraints}}$$

### 1. Structure Energy (Local Coherence)

Prevents "noise" by encouraging tiles to match their neighbors:

$$E_{\text{structure}} = -J \sum_{\langle i,j \rangle} s_i s_j$$

Where:
- \( \langle i,j \rangle \) denotes neighboring tiles
- \( J > 0 \): neighboring tiles prefer same state (creates rooms/corridors)
- This is the classic **Ising model** interaction

**Effect**: Creates contiguous regions (rooms) rather than salt-and-pepper noise.

### 2. Bias Energy (Boundary & Density)

Controls overall floor/wall ratio and enforces boundaries:

$$E_{\text{bias}} = -\sum_i h_i s_i$$

Where:
- \( h_i < 0 \) for edge tiles (prefer walls → natural boundaries)
- \( h_i > 0 \) for interior (prefer floors → open space)
- \( h_i \approx 0 \) for flexibility

**Effect**: Walls form naturally at edges, interior stays open.

### 3. Connectivity Energy (Graph Reachability)

The hardest part! We want all floor tiles to be connected. Options:

**Option A: Soft Penalty (Simple)**
```
Penalize isolated floor regions by approximating connectivity
through local checks (e.g., penalize floor tiles surrounded by walls)
```

**Option B: Explicit Constraint (Advanced)**
```
Use auxiliary variables to represent connected components.
This requires more sophisticated PGM techniques (factor graphs).
```

**Option C: Post-Processing (Pragmatic)**
```
Sample first, then run flood-fill and resample if disconnected.
Works well with annealing schedules.
```

### 4. Aesthetic Energy (Optional Flourishes)

Encode subjective preferences:

$$E_{\text{room}} = -\alpha \cdot (\text{# of rectangular regions})$$
$$E_{\text{corridor}} = -\gamma \cdot (\text{# of 1-tile-wide paths})$$

These require more complex non-local interactions.

---

## Node Types & States

### Phase 1: Binary Tiles (Ising Model)

Each tile is a **SpinNode**:
- State: `bool` (True = floor, False = wall)
- JAX representation: `jnp.bool_` array

Simple but effective for basic dungeons.

### Phase 2: Categorical Tiles (Potts Model)

Each tile is a **CategoricalNode**:
- States: {wall, floor, door, corridor, treasure, spawn, ...}
- JAX representation: `jnp.uint8` (one-hot or integer encoding)

Allows richer dungeons with typed features.

### Phase 3: Continuous Features

Mix discrete tiles with continuous attributes:
- Elevation/height (for 3D dungeons)
- Danger level (for enemy placement)
- Treasure density

Requires custom node types extending `AbstractNode`.

---

## Sampling Strategy

### Block Gibbs Sampling

THRML's core algorithm. For a 2D grid:

1. **Graph Coloring**: Partition tiles into independent sets
   - Checkerboard pattern: even tiles, odd tiles (2 colors)
   - No two adjacent tiles in same color
   
2. **Parallel Updates**: Sample all tiles in a color simultaneously
   ```python
   for color in [even_tiles, odd_tiles]:
       # All tiles in 'color' can update in parallel (GPU-friendly!)
       sample_block(color, given_state_of_other_tiles)
   ```

3. **Repeat**: Iterate until convergence (or for fixed steps)

**Why it works**: 
- Tiles in same color don't interact → truly parallel
- JAX compiles this to efficient GPU kernels
- On Extropic hardware, sampling step is O(1) in wall-clock time!

### Annealing Schedule

Start with high temperature (random), cool down:

```python
beta_schedule = [0.1, 0.5, 1.0, 2.0, 5.0]  # increasing beta

for beta in beta_schedule:
    model.beta = beta
    run_gibbs_for_N_steps(model)
```

High temp → explore broadly  
Low temp → refine to local optimum

**Analogy**: Like cooling molten metal to form crystals.

---

## Implementation Phases

### Phase 1: Minimal Viable Dungeon ✅ (Quick Start)

```python
- Grid of SpinNodes
- Ising interactions (neighbor matching)
- Edge biases (boundary walls)
- Checkerboard block Gibbs sampling
- Matplotlib visualization
```

**Status**: `entropit_quickstart.py` implements this!

### Phase 2: Connectivity Constraints

```python
- Add soft connectivity penalties
- Implement flood-fill check
- Resample if disconnected
- Visualize connected components
```

### Phase 3: Interactive UI

```python
- Gradio interface
- User paints constraints (fixed walls/floors)
- Clamped blocks in THRML
- Real-time generation
- Slider for parameters (β, room size, etc.)
```

### Phase 4: Advanced Features

```python
- Categorical nodes (doors, treasure, enemies)
- Multi-floor dungeons (3D)
- Path length constraints (spawn → exit distance)
- Style transfer (learn from existing dungeons)
```

---

## THRML Mapping

### Concepts

| Dungeon Concept | THRML Object | Purpose |
|----------------|--------------|---------|
| Tile | `SpinNode` / `CategoricalNode` | Represents state of one tile |
| Grid Adjacency | `edges` in `IsingEBM` | Defines which tiles interact |
| Room Preference | `biases` parameter | Per-tile bias toward floor/wall |
| Matching Neighbors | `weights` parameter | Strength of neighbor interactions |
| Sampling Round | `Block` | Group of tiles updated together |
| Generation Process | `IsingSamplingProgram` | Full sampling specification |

### Code Structure

```python
# Define the graph
nodes = [[SpinNode() for _ in row] for row in grid]
edges = [...] # adjacency

# Define energy parameters
biases = jnp.array([...])  # per-tile preferences
weights = jnp.array([...]) # per-edge interaction strength
beta = 2.0  # temperature

# Create model
model = IsingEBM(nodes, edges, biases, weights, beta)

# Define sampling blocks (checkerboard)
even_block = Block([tiles where (i+j) % 2 == 0])
odd_block = Block([tiles where (i+j) % 2 == 1])

# Create sampling program
program = IsingSamplingProgram(model, [even_block, odd_block], [])

# Sample!
samples = sample_states(key, program, schedule, init_state, ...)
```

---

## Performance Considerations

### Current (GPU)

- Grid size: Up to ~100×100 with good performance
- Sample time: ~1-10 seconds depending on schedule
- Bottleneck: Python overhead, JAX compilation

### With Extropic Hardware (Simulated)

- Grid size: 1000×1000+ (scales with chip size)
- Sample time: Milliseconds (physical sampling time)
- Bottleneck: I/O to/from chip

### Optimization Strategies

1. **JIT Compilation**: Wrap sampling in `@jax.jit` (THRML does this)
2. **Batch Generation**: Generate 100s of dungeons in parallel
3. **Smart Initialization**: Start from good guess (not random)
4. **Adaptive Annealing**: Stop early if converged

---

## Extensions & Research Directions

### 1. Metroidvania Constraints

Model keys and locked doors as constraint satisfaction:
- Door nodes depend on player having key
- Energy penalizes unreachable treasure
- Results in dungeons with intended progression

### 2. Procedural Narratives

Add "story nodes" that chain together:
- Boss room must be far from spawn
- Shop must be near safe area
- Lore items scattered with pattern

### 3. Multi-Agent Design

Multiple designers paint constraints, PGM finds compromise:
- Designer A wants challenging
- Designer B wants explorable
- Sample finds dungeon satisfying both

### 4. Real-Time Editing

As player explores, unexplored regions resample:
- Clamped: tiles player has seen
- Free: tiles beyond fog of war
- Creates infinite, coherent worlds

---

## Mathematical Depth

For those interested in the theory:

### Why Boltzmann Distribution?

The Boltzmann distribution is the **maximum entropy distribution** subject to energy constraints. This means:
- It's the "least biased" distribution given our preferences
- Avoids overfitting to arbitrary design choices
- Naturally balances competing goals

### Connection to Physics

The Ising model was originally developed for ferromagnetism:
- Spins align → magnets
- Tiles match → rooms

Extropic's insight: Use *actual physics* (thermodynamics of stochastic circuits) to sample these distributions in hardware!

### Why Sampling is Hard

Computing \( Z = \sum_s e^{-\beta E(s)} \) is #P-hard (counting problem).  
Sampling from \( P(s) \) is generally hard too.  
But **thermodynamic systems do this naturally** by equilibrating!

---

## Next Steps

1. **Run the Quick Start**: `python entropit_quickstart.py`
2. **Read the Code**: Understand how THRML concepts map to implementation
3. **Tweak Parameters**: Change `biases`, `weights`, `beta` and see what happens
4. **Add Constraints**: Start simple (fix spawn tile), then grow
5. **Build the UI**: Gradio makes it easy to add interactivity

---

**Welcome to probabilistic dungeon generation.** 🔥

*Let physics design your levels.*

