"""
EntroPit Quick Start
====================
Minimal working example of probabilistic dungeon generation using THRML.

This script demonstrates:
- Setting up an Ising model for 2D grid generation
- Block Gibbs sampling with checkerboard partitioning
- Energy-based constraint specification
- Visualization of sampled dungeons

For technical details, see ARCHITECTURE.md
For getting started guide, see README.md
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init


def create_simple_dungeon(grid_size=10, seed=42):
    """
    Generate a dungeon using an Ising model.
    
    Model:
    - Each tile is a SpinNode (True = floor, False = wall)
    - Energy favors neighboring tiles with matching states (creates rooms)
    - Edge tiles biased toward walls (natural boundaries)
    
    Args:
        grid_size: Width and height of dungeon grid
        seed: Random seed for reproducibility
        
    Returns:
        dungeons: Array of sampled configurations [n_samples, grid_size, grid_size]
        nodes: 2D list of SpinNode objects
    """
    print(f"🏰 Generating {grid_size}x{grid_size} dungeon with EntroPit...")
    
    # Step 1: Create the graph structure
    # ===================================
    nodes = [[SpinNode() for _ in range(grid_size)] for _ in range(grid_size)]
    flat_nodes = [node for row in nodes for node in row]
    
    # Define edges (4-connected grid = each tile has up to 4 neighbors)
    edges = []
    for i in range(grid_size):
        for j in range(grid_size):
            # Right neighbor
            if j < grid_size - 1:
                edges.append((nodes[i][j], nodes[i][j+1]))
            # Down neighbor
            if i < grid_size - 1:
                edges.append((nodes[i][j], nodes[i+1][j]))
    
    # Step 2: Define the energy function
    # ===================================
    # Energy = -(weights · neighbor_matches) - (biases · tile_states)
    # Lower energy = higher probability
    
    # BIASES: Per-tile preferences
    # Negative bias → prefers False (wall)
    # Positive bias → prefers True (floor)
    biases = []
    for i in range(grid_size):
        for j in range(grid_size):
            is_edge = (i == 0 or i == grid_size-1 or j == 0 or j == grid_size-1)
            # Try tweaking these values! (-5.0 = very thick walls, 2.0 = very open)
            biases.append(-2.0 if is_edge else 0.5)
    
    biases = jnp.array(biases)
    
    # WEIGHTS: Pairwise interaction strength
    # Positive weight → neighbors prefer same state (creates rooms/corridors)
    # Try tweaking: 1.5 = bigger rooms, 0.3 = more fragmented
    weights = jnp.ones(len(edges)) * 0.8
    
    # BETA: Inverse temperature (higher = more structured, lower = more random)
    # Try tweaking: 0.5 = chaotic, 5.0 = very structured
    beta = jnp.array(2.0)
    
    # Create the Ising model (this compiles the energy function)
    model = IsingEBM(flat_nodes, edges, biases, weights, beta)
    
    # Step 3: Set up block Gibbs sampling
    # ====================================
    # Checkerboard coloring: tiles in same color don't share edges
    # → can be sampled in parallel (GPU-friendly!)
    
    even_nodes = [nodes[i][j] for i in range(grid_size) for j in range(grid_size) 
                  if (i + j) % 2 == 0]
    odd_nodes = [nodes[i][j] for i in range(grid_size) for j in range(grid_size) 
                 if (i + j) % 2 == 1]
    
    free_blocks = [Block(even_nodes), Block(odd_nodes)]
    program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
    
    # Step 4: Initialize and run sampling
    # ===================================
    key = jax.random.key(seed)
    k_init, k_samp = jax.random.split(key, 2)
    
    # Hinton initialization: balanced starting point
    init_state = hinton_init(k_init, model, free_blocks, ())
    
    # Sampling schedule
    schedule = SamplingSchedule(
        n_warmup=200,       # Equilibration steps (increase for better quality)
        n_samples=10,       # Number of independent dungeons to generate
        steps_per_sample=20 # Steps between samples (for decorrelation)
    )
    
    print("🔥 Running Gibbs sampling...")
    samples = sample_states(k_samp, program, schedule, init_state, [], [Block(flat_nodes)])
    
    # Step 5: Extract and reshape results
    # ====================================
    # samples[0] has shape (n_samples, n_nodes) - reshape to grid
    dungeons = samples[0].reshape(schedule.n_samples, grid_size, grid_size)
    
    print(f"✨ Generated {schedule.n_samples} dungeon variants!")
    return dungeons, nodes


def visualize_dungeons(dungeons, n_show=4):
    """
    Render and save dungeon visualizations.
    
    Args:
        dungeons: Array of dungeon grids [n_dungeons, height, width]
        n_show: Number of dungeons to display (max 4 for readability)
    """
    n_show = min(n_show, len(dungeons))
    fig, axes = plt.subplots(1, n_show, figsize=(4*n_show, 4))
    
    if n_show == 1:
        axes = [axes]
    
    # False (0) = wall (dark), True (1) = floor (light)
    cmap = ListedColormap(['#2c3e50', '#ecf0f1'])
    
    for idx, ax in enumerate(axes):
        dungeon = dungeons[idx].astype(int)
        ax.imshow(dungeon, cmap=cmap, interpolation='nearest')
        ax.set_title(f'Dungeon #{idx+1}')
        ax.axis('off')
        
        # Add subtle grid lines
        ax.set_xticks(jnp.arange(-0.5, len(dungeon), 1), minor=True)
        ax.set_yticks(jnp.arange(-0.5, len(dungeon), 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('entropit_dungeons.png', dpi=150, bbox_inches='tight')
    print("💾 Saved visualization to 'entropit_dungeons.png'")
    plt.show()


def analyze_dungeon(dungeon):
    """
    Print statistics about a generated dungeon.
    
    Args:
        dungeon: Boolean array [height, width] where True = floor
    """
    floor_tiles = jnp.sum(dungeon)
    total_tiles = dungeon.size
    wall_tiles = total_tiles - floor_tiles
    
    print(f"\n📊 Dungeon Statistics:")
    print(f"  • Size: {dungeon.shape[0]}x{dungeon.shape[1]}")
    print(f"  • Floor tiles: {floor_tiles} ({100*floor_tiles/total_tiles:.1f}%)")
    print(f"  • Wall tiles: {wall_tiles} ({100*wall_tiles/total_tiles:.1f}%)")
    
    # Note: For production, you'd want to check connectivity using NetworkX
    # to ensure all floor tiles are reachable
    

if __name__ == "__main__":
    print("=" * 60)
    print("🌀 EntroPit - Probabilistic Dungeon Generator")
    print("   Powered by THRML & Thermodynamic Computing")
    print("=" * 60)
    print()
    
    # Generate dungeons using Gibbs sampling
    dungeons, nodes = create_simple_dungeon(grid_size=12, seed=42)
    
    # Visualize the results
    visualize_dungeons(dungeons, n_show=4)
    
    # Print statistics
    analyze_dungeon(dungeons[0])
    
    print()
    print("=" * 60)
    print("🎉 Success! Your probabilistic dungeon generator works!")
    print()
    print("Experiment:")
    print("  • Modify parameters in create_simple_dungeon()")
    print("  • Try different grid_size values (8, 16, 24, 32)")
    print("  • Adjust beta (temperature), biases, weights")
    print()
    print("Learn more:")
    print("  • README.md - Project overview and getting started")
    print("  • ARCHITECTURE.md - Mathematical formulation")
    print("=" * 60)

