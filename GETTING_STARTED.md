# 🚀 Getting Started with EntroPit

This guide will walk you through running your first probabilistic dungeon generation in under 5 minutes.

---

## Step 1: Set Up Your Environment

### Create a Project Directory

```bash
mkdir entropit-project
cd entropit-project
```

### Install Dependencies

You have two options:

**Option A: Using pip directly**
```bash
pip install jax jaxlib
pip install thrml
pip install matplotlib pillow numpy
```

**Option B: Using the requirements file**

If you have `entropit_requirements.txt`:
```bash
pip install -r entropit_requirements.txt
```

### Verify Installation

```bash
python -c "import thrml; print(f'THRML version: {thrml.__version__}')"
python -c "import jax; print(f'JAX devices: {jax.devices()}')"
```

You should see:
```
THRML version: 0.1.3
JAX devices: [CpuDevice(id=0)]  # or GpuDevice if you have CUDA
```

---

## Step 2: Run Your First Dungeon

Copy the quick start script to your project directory (or create it):

```bash
# If you have entropit_quickstart.py already
python entropit_quickstart.py
```

### What You'll See

```
============================================================
🌀 EntroPit - Probabilistic Dungeon Generator
   Powered by THRML & Thermodynamic Computing
============================================================

🏰 Generating 12x12 dungeon with EntroPit...
🔥 Running Gibbs sampling...
✨ Generated 10 dungeon variants!
💾 Saved visualization to 'entropit_dungeons.png'

📊 Dungeon Statistics:
  • Size: 12x12
  • Floor tiles: 68 (47.2%)
  • Wall tiles: 76 (52.8%)

============================================================
🎉 Success! Your probabilistic dungeon generator works!

Next steps:
  1. Tweak biases/weights in create_simple_dungeon()
  2. Add connectivity constraints
  3. Build the interactive Gradio UI
  4. Add treasure/enemy placement
============================================================
```

### View Your Dungeons

Open `entropit_dungeons.png` to see 4 generated dungeon variants!

---

## Step 3: Understand What Just Happened

### The Model

Each tile in the grid is modeled as a **SpinNode**:
- `True` (1) = Floor tile (walkable)
- `False` (0) = Wall tile (blocked)

### The Energy Function

The Ising model creates an energy landscape:

```python
E(dungeon) = -Σ(edge_weights × neighbor_matches) - Σ(biases × tile_states)
```

**Lower energy = better dungeon**

### The Sampling Process

1. **Initialize**: Random or biased starting configuration
2. **Iterate**: Update tiles using Gibbs sampling (checkerboard pattern)
3. **Converge**: System equilibrates to low-energy (high-quality) dungeon
4. **Sample**: Collect final configuration(s)

### Why It's Probabilistic

Each run explores a different path through the energy landscape, giving you varied dungeons that all satisfy your constraints!

---

## Step 4: Experiment!

### Tweak the Parameters

Open `entropit_quickstart.py` and modify:

#### 1. Grid Size
```python
dungeons, nodes = create_simple_dungeon(grid_size=20, seed=42)
```
Try: 8, 16, 24, 32

#### 2. Bias Strength (Line ~50)
```python
# More negative = prefer walls, more positive = prefer floors
biases.append(-2.0 if is_edge else 0.5)
```
Try: `-5.0` for thick walls, `2.0` for open spaces

#### 3. Coupling Strength (Line ~55)
```python
weights = jnp.ones(len(edges)) * 0.8
```
Try: `1.5` for bigger rooms, `0.3` for noisy/fragmented

#### 4. Temperature (Line ~58)
```python
beta = jnp.array(2.0)
```
Try: `0.5` for chaos, `5.0` for very structured

#### 5. Sampling Schedule (Line ~77)
```python
schedule = SamplingSchedule(
    n_warmup=200,      # Equilibration steps
    n_samples=10,      # Number of dungeons
    steps_per_sample=20  # Steps between samples
)
```
Try: `n_warmup=1000` for better convergence, `n_samples=50` for variety

### Observe the Changes

After each modification:
```bash
python entropit_quickstart.py
```

See how the dungeons change!

---

## Step 5: Add Your First Constraint

Let's force a spawn point in the center.

### Modify the Code

Add this after line ~75 (after creating `free_blocks`):

```python
# Force center tile to be a floor (spawn point)
center_idx = grid_size // 2
spawn_node = nodes[center_idx][center_idx]

# Create a clamped block (fixed during sampling)
clamped_blocks = [Block([spawn_node])]
clamp_vals = [jnp.array([True], dtype=jnp.bool_)]  # True = floor

# Update program creation (line ~76)
program = IsingSamplingProgram(model, free_blocks, clamped_blocks)

# Update sampling call (line ~84) to use clamp_vals
samples = sample_states(
    k_samp, program, schedule, init_state, 
    clamp_vals,  # Add this
    [Block(flat_nodes)]
)
```

Now the center will always be a floor tile (your spawn point)!

---

## Step 6: Visualize the Sampling Process

Let's see how the dungeon evolves during sampling.

### Create a New Script: `watch_sampling.py`

```python
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

from thrml import SpinNode, Block, SamplingSchedule, sample_with_observation
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init
from thrml.observers import StateObserver

# ... (copy the create_simple_dungeon function, but modify schedule)

schedule = SamplingSchedule(
    n_warmup=0,
    n_samples=100,  # More samples to see evolution
    steps_per_sample=1  # Every step
)

# Use sample_with_observation to see intermediate states
observer = StateObserver([Block(flat_nodes)])
carry_init = observer.init()

_, samples = sample_with_observation(
    k_samp, program, schedule, init_state, [],
    carry_init, observer
)

# Animate the evolution
fig, ax = plt.subplots()
cmap = ListedColormap(['#2c3e50', '#ecf0f1'])

def update(frame):
    ax.clear()
    dungeon = samples[0][frame].reshape(grid_size, grid_size)
    ax.imshow(dungeon, cmap=cmap, interpolation='nearest')
    ax.set_title(f'Sampling Step {frame}')
    ax.axis('off')

anim = FuncAnimation(fig, update, frames=100, interval=50)
plt.show()
```

Watch your dungeon emerge from chaos!

---

## Step 7: Common Issues & Solutions

### Issue: Sampling is slow

**Solution**: Reduce grid size or warmup steps while testing
```python
create_simple_dungeon(grid_size=8, seed=42)  # Smaller grid
schedule = SamplingSchedule(n_warmup=50, ...)  # Fewer steps
```

### Issue: Dungeons look too random

**Solution**: Increase temperature (beta) or coupling strength
```python
beta = jnp.array(5.0)  # More structured
weights = jnp.ones(len(edges)) * 1.5  # Stronger clustering
```

### Issue: Dungeons are all walls/all floors

**Solution**: Balance biases
```python
# Check that biases aren't too extreme
biases.append(-1.0 if is_edge else 0.5)  # More balanced
```

### Issue: JAX compilation is slow on first run

**Normal!** JAX traces and compiles on first call. Subsequent runs are fast.

---

## Step 8: Next Steps

### Build the Interactive UI

Create `entropit_ui.py`:

```python
import gradio as gr
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

def generate_dungeon(grid_size, beta, edge_bias, seed):
    # ... call your dungeon generation code
    # ... return PIL Image
    pass

demo = gr.Interface(
    fn=generate_dungeon,
    inputs=[
        gr.Slider(8, 32, value=12, step=4, label="Grid Size"),
        gr.Slider(0.5, 5.0, value=2.0, step=0.5, label="Temperature (β)"),
        gr.Slider(-5.0, 0.0, value=-2.0, step=0.5, label="Edge Bias"),
        gr.Number(value=42, label="Seed")
    ],
    outputs=gr.Image(label="Generated Dungeon"),
    title="🏰 EntroPit - Probabilistic Dungeon Generator",
    description="Generate dungeons using Gibbs sampling on an Ising model"
)

demo.launch()
```

Then run:
```bash
python entropit_ui.py
```

### Add Connectivity Checking

Use NetworkX to verify all floors are reachable:

```python
import networkx as nx

def check_connectivity(dungeon):
    G = nx.grid_2d_graph(*dungeon.shape)
    floor_nodes = [(i,j) for i in range(dungeon.shape[0]) 
                   for j in range(dungeon.shape[1]) if dungeon[i,j]]
    subgraph = G.subgraph(floor_nodes)
    return nx.is_connected(subgraph)
```

### Implement Treasure Placement

Add a second Ising model for treasure density, conditioned on floor tiles.

### Export to Game Engine

Save as JSON for Unity/Godot:
```python
import json

dungeon_data = {
    "width": grid_size,
    "height": grid_size,
    "tiles": dungeon.tolist(),
    "spawn": [grid_size//2, grid_size//2]
}

with open('dungeon.json', 'w') as f:
    json.dump(dungeon_data, f)
```

---

## Resources

- **THRML Documentation**: https://docs.thrml.ai/
- **Architecture Deep-Dive**: See `ENTROPIT_ARCHITECTURE.md`
- **JAX Quickstart**: https://jax.readthedocs.io/en/latest/quickstart.html
- **Ising Model Tutorial**: https://en.wikipedia.org/wiki/Ising_model

---

## Get Help

If you run into issues:

1. Check that all dependencies are installed
2. Verify JAX is working: `python -c "import jax; print(jax.devices())"`
3. Start with smallest possible example (grid_size=8, n_warmup=50)
4. Read error messages carefully - JAX errors can be cryptic but informative

---

## Have Fun!

You're now generating dungeons using the same probabilistic principles that will power Extropic's thermodynamic computers.

**Each dungeon is a sample from a probability distribution defined by physics.** 🔥

Happy dungeon crafting! ⚔️🏰

---

*Questions or cool results? Share them with the THRML community!*

