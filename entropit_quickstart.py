"""
EntroPit - Quick Start Example
A minimal dungeon generator to verify THRML is working
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init


def create_simple_dungeon(grid_size=10, seed=42):
    """
    Create a simple dungeon using an Ising model.
    
    In this minimal example:
    - Each tile is a SpinNode (True = floor, False = wall)
    - Neighboring tiles prefer to match (creates rooms)
    - Edge tiles are biased toward walls (natural boundaries)
    """
    print(f"🏰 Generating {grid_size}x{grid_size} dungeon with EntroPit...")
    
    # Create grid of nodes
    nodes = [[SpinNode() for _ in range(grid_size)] for _ in range(grid_size)]
    flat_nodes = [node for row in nodes for node in row]
    
    # Create edges (4-connected grid)
    edges = []
    for i in range(grid_size):
        for j in range(grid_size):
            # Right neighbor
            if j < grid_size - 1:
                edges.append((nodes[i][j], nodes[i][j+1]))
            # Down neighbor
            if i < grid_size - 1:
                edges.append((nodes[i][j], nodes[i+1][j]))
    
    # Biases: encourage floor tiles in interior, walls on edges
    biases = []
    for i in range(grid_size):
        for j in range(grid_size):
            is_edge = (i == 0 or i == grid_size-1 or j == 0 or j == grid_size-1)
            # Positive bias = prefer True (floor), negative = prefer False (wall)
            biases.append(-2.0 if is_edge else 0.5)
    
    biases = jnp.array(biases)
    
    # Weights: positive = same state preferred (creates clusters)
    weights = jnp.ones(len(edges)) * 0.8
    
    # Temperature parameter (lower = more structure)
    beta = jnp.array(2.0)
    
    # Create the Ising model
    model = IsingEBM(flat_nodes, edges, biases, weights, beta)
    
    # Set up sampling: checkerboard coloring for parallel updates
    # Even tiles
    even_nodes = [nodes[i][j] for i in range(grid_size) for j in range(grid_size) 
                  if (i + j) % 2 == 0]
    # Odd tiles
    odd_nodes = [nodes[i][j] for i in range(grid_size) for j in range(grid_size) 
                 if (i + j) % 2 == 1]
    
    free_blocks = [Block(even_nodes), Block(odd_nodes)]
    program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
    
    # Initialize and sample
    key = jax.random.key(seed)
    k_init, k_samp = jax.random.split(key, 2)
    init_state = hinton_init(k_init, model, free_blocks, ())
    
    # Sampling schedule: warmup to equilibrate, then collect samples
    schedule = SamplingSchedule(
        n_warmup=200,      # Let it converge
        n_samples=10,      # Get multiple dungeon variants
        steps_per_sample=20 # Space them out
    )
    
    print("🔥 Running Gibbs sampling...")
    samples = sample_states(k_samp, program, schedule, init_state, [], [Block(flat_nodes)])
    
    # Samples shape: (n_samples, n_nodes)
    # Reshape to grid for visualization
    dungeons = samples[0].reshape(schedule.n_samples, grid_size, grid_size)
    
    print(f"✨ Generated {schedule.n_samples} dungeon variants!")
    return dungeons, nodes


def visualize_dungeons(dungeons, n_show=4):
    """Show a grid of generated dungeons"""
    n_show = min(n_show, len(dungeons))
    fig, axes = plt.subplots(1, n_show, figsize=(4*n_show, 4))
    
    if n_show == 1:
        axes = [axes]
    
    # Custom colormap: walls are dark, floors are light
    cmap = ListedColormap(['#2c3e50', '#ecf0f1'])  # dark blue-gray, light gray
    
    for idx, ax in enumerate(axes):
        dungeon = dungeons[idx].astype(int)
        ax.imshow(dungeon, cmap=cmap, interpolation='nearest')
        ax.set_title(f'Dungeon #{idx+1}')
        ax.axis('off')
        
        # Add grid lines
        ax.set_xticks(jnp.arange(-0.5, len(dungeon), 1), minor=True)
        ax.set_yticks(jnp.arange(-0.5, len(dungeon), 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('entropit_dungeons.png', dpi=150, bbox_inches='tight')
    print("💾 Saved visualization to 'entropit_dungeons.png'")
    plt.show()


def analyze_dungeon(dungeon):
    """Print some statistics about the generated dungeon"""
    floor_tiles = jnp.sum(dungeon)
    total_tiles = dungeon.size
    wall_tiles = total_tiles - floor_tiles
    
    print(f"\n📊 Dungeon Statistics:")
    print(f"  • Size: {dungeon.shape[0]}x{dungeon.shape[1]}")
    print(f"  • Floor tiles: {floor_tiles} ({100*floor_tiles/total_tiles:.1f}%)")
    print(f"  • Wall tiles: {wall_tiles} ({100*wall_tiles/total_tiles:.1f}%)")
    
    # Count "rooms" as connected components of floor tiles
    # (simplified - just check largest cluster size)
    # For a real implementation, we'd use proper connected component analysis
    

if __name__ == "__main__":
    print("=" * 60)
    print("🌀 EntroPit - Probabilistic Dungeon Generator")
    print("   Powered by THRML & Thermodynamic Computing")
    print("=" * 60)
    print()
    
    # Generate dungeons
    dungeons, nodes = create_simple_dungeon(grid_size=12, seed=42)
    
    # Show some examples
    visualize_dungeons(dungeons, n_show=4)
    
    # Analyze the first one
    analyze_dungeon(dungeons[0])
    
    print()
    print("=" * 60)
    print("🎉 Success! Your probabilistic dungeon generator works!")
    print()
    print("Next steps:")
    print("  1. Tweak biases/weights in create_simple_dungeon()")
    print("  2. Add connectivity constraints")
    print("  3. Build the interactive Gradio UI")
    print("  4. Add treasure/enemy placement")
    print("=" * 60)

