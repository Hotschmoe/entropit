# Getting Started with EntroPit

Welcome to EntroPit! This guide will help you get up and running quickly.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/entropit.git
cd entropit
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) GPU Support

For faster THRML generation with GPU acceleration:

```bash
# For CUDA 12.x
pip install jax[cuda12]

# For CUDA 11.x
pip install jax[cuda11]
```

### 4. Verify Installation

```bash
python examples/quickstart.py
```

You should see dungeon generation output and a saved image in `output/quickstart_dungeons.png`.

---

## Quick Examples

### Example 1: Generate a Single Dungeon

```python
from entropit import generate_thrml, analyze_dungeon

# Generate one dungeon using THRML
dungeons, metadata = generate_thrml(grid_size=16, beta=2.0, seed=42)

# Analyze the first dungeon
metrics = analyze_dungeon(dungeons[0])

print(f"Connected: {metrics['is_connected']}")
print(f"Floor coverage: {metrics['floor_ratio']*100:.1f}%")
```

### Example 2: Compare Multiple Methods

```python
from entropit import generate_thrml, generate_traditional

# Generate with THRML
thrml_dungeon, _ = generate_thrml(grid_size=24, seed=42)

# Generate with Cellular Automata
ca_dungeon, _ = generate_traditional("cellular_automata", grid_size=24, seed=42)

# Generate with BSP
bsp_dungeon, _ = generate_traditional("bsp", grid_size=24, seed=42)
```

### Example 3: Adjust THRML Parameters

```python
from entropit import generate_thrml

# Very structured dungeon (high temperature)
structured, _ = generate_thrml(grid_size=16, beta=5.0, edge_bias=-3.0)

# More random dungeon (low temperature)
random_like, _ = generate_thrml(grid_size=16, beta=0.5, edge_bias=-1.0)

# Big open rooms (high coupling)
open_rooms, _ = generate_thrml(grid_size=16, beta=2.0, coupling=1.5)

# Narrow corridors (low coupling)
corridors, _ = generate_thrml(grid_size=16, beta=2.0, coupling=0.3)
```

---

## Running the Examples

### Quickstart Demo

```bash
python examples/quickstart.py
```

Generates 4 dungeon variants and saves visualization to `output/`.

### Comparison Demo

```bash
python examples/comparison_demo.py
```

Side-by-side comparison of all 5 generation methods.

### Benchmark Demo

```bash
python examples/benchmark_demo.py
```

Runs 10 iterations of each method and computes statistics. Takes 1-2 minutes.

### Interactive Web UI

```bash
python examples/interactive_ui.py
```

Launches Gradio interface at `http://localhost:7860` for live parameter tuning.

---

## Understanding the Parameters

### THRML Parameters

| Parameter | Range | Effect |
|-----------|-------|--------|
| `grid_size` | 8-32 | Size of the dungeon |
| `beta` | 0.5-5.0 | Higher = more structured, lower = more random |
| `edge_bias` | -5.0 to 0.0 | More negative = thicker walls at boundaries |
| `coupling` | 0.3-2.0 | Higher = bigger open rooms, lower = narrow corridors |
| `n_warmup` | 100-500 | More warmup = better quality (but slower) |
| `n_samples` | 1-10 | Number of independent dungeons to generate |

### Traditional Method Parameters

**Random:**
- `floor_probability`: 0-1 (default: 0.5)

**Cellular Automata:**
- `initial_floor_prob`: 0-1 (default: 0.45)
- `iterations`: 1-10 (default: 5)

**BSP:**
- `min_room_size`: 3-10 (default: 4)
- `max_room_size`: 8-15 (default: 10)

**Drunkard's Walk:**
- `floor_percentage`: 0-1 (default: 0.4)

---

## Next Steps

1. **Experiment** with parameters in `examples/quickstart.py`
2. **Read** `docs/ARCHITECTURE.md` for technical details
3. **Explore** the interactive UI for real-time parameter tuning
4. **Benchmark** to see the THRML vs traditional tradeoffs

---

## Troubleshooting

### ImportError: No module named 'entropit'

Make sure you're running from the project root:
```bash
cd /path/to/entropit
python examples/quickstart.py
```

Or install the package in development mode:
```bash
pip install -e .
```

### JAX/THRML Issues

If you encounter JAX errors, try:
```bash
pip install --upgrade jax jaxlib thrml
```

### Windows Encoding Issues

The examples automatically handle Windows console encoding, but if you see garbled characters, run:
```bash
chcp 65001  # Set console to UTF-8
```

### Slow Generation

- THRML is inherently slower than traditional methods (that's the hardware tradeoff!)
- Reduce `grid_size` (e.g., 12 instead of 24)
- Reduce `n_warmup` (e.g., 100 instead of 200)
- Use GPU support for 2-5x speedup

---

## Getting Help

- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions
- **Documentation**: Check `docs/` directory
- **Examples**: Study the `examples/` directory

Happy dungeon generating! 🏰

