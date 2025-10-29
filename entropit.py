"""
EntroPit - Main Module
======================
Core dungeon generation functions using THRML and traditional methods.

This module provides the main API for dungeon generation:
- generate_thrml(): THRML-based probabilistic generation
- generate_traditional(): Classical algorithmic generation
"""

import numpy as np
import jax.numpy as jnp
from typing import Tuple, Dict
import time

# Import THRML components
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init
import jax

# Import traditional generators
import traditional_generators as trad
from benchmark import check_connectivity, calculate_playability_score


def generate_thrml(
    grid_size: int = 12,
    beta: float = 2.0,
    edge_bias: float = -2.0,
    coupling: float = 0.8,
    n_warmup: int = 200,
    n_samples: int = 1,
    steps_per_sample: int = 20,
    seed: int = None,
    verbose: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Generate dungeon(s) using THRML Ising model.
    
    Args:
        grid_size: Width and height of dungeon grid
        beta: Inverse temperature (higher = more structured)
        edge_bias: Bias for edge tiles (negative = prefer walls)
        coupling: Neighbor interaction strength (higher = bigger rooms)
        n_warmup: Equilibration steps before sampling
        n_samples: Number of dungeon variants to generate
        steps_per_sample: Steps between samples (for decorrelation)
        seed: Random seed for reproducibility
        verbose: Print generation progress
        
    Returns:
        dungeons: Array [n_samples, grid_size, grid_size] of boolean arrays (True = floor)
        metadata: Dict with generation info (time, parameters, etc.)
    """
    if seed is None:
        import random
        seed = random.randint(0, 999999)
    
    start_time = time.time()
    
    if verbose:
        print(f"[*] Generating {grid_size}x{grid_size} dungeon with THRML (Ising model)")
        print(f"    Parameters: beta={beta}, edge_bias={edge_bias}, coupling={coupling}")
        print(f"    Seed: {seed}")
    
    # Step 1: Create graph structure
    nodes = [[SpinNode() for _ in range(grid_size)] for _ in range(grid_size)]
    flat_nodes = [node for row in nodes for node in row]
    
    # Create edges (4-connected grid)
    edges = []
    for i in range(grid_size):
        for j in range(grid_size):
            if j < grid_size - 1:
                edges.append((nodes[i][j], nodes[i][j+1]))
            if i < grid_size - 1:
                edges.append((nodes[i][j], nodes[i+1][j]))
    
    # Step 2: Define energy function
    biases = []
    for i in range(grid_size):
        for j in range(grid_size):
            is_edge = (i == 0 or i == grid_size-1 or j == 0 or j == grid_size-1)
            biases.append(edge_bias if is_edge else -edge_bias/4)  # Interior has opposite bias
    
    biases = jnp.array(biases)
    weights = jnp.ones(len(edges)) * coupling
    beta_jax = jnp.array(beta)
    
    # Create Ising model
    model = IsingEBM(flat_nodes, edges, biases, weights, beta_jax)
    
    # Step 3: Block Gibbs sampling setup (checkerboard)
    even_nodes = [nodes[i][j] for i in range(grid_size) for j in range(grid_size) 
                  if (i + j) % 2 == 0]
    odd_nodes = [nodes[i][j] for i in range(grid_size) for j in range(grid_size) 
                 if (i + j) % 2 == 1]
    
    free_blocks = [Block(even_nodes), Block(odd_nodes)]
    program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
    
    # Step 4: Initialize and sample
    key = jax.random.key(seed)
    k_init, k_samp = jax.random.split(key, 2)
    init_state = hinton_init(k_init, model, free_blocks, ())
    
    schedule = SamplingSchedule(
        n_warmup=n_warmup,
        n_samples=n_samples,
        steps_per_sample=steps_per_sample
    )
    
    if verbose:
        print(f"[*] Running Gibbs sampling ({n_warmup} warmup + {n_samples} samples)...")
    
    samples = sample_states(k_samp, program, schedule, init_state, [], [Block(flat_nodes)])
    
    # Extract and reshape
    dungeons_jax = samples[0].reshape(n_samples, grid_size, grid_size)
    dungeons = np.array(dungeons_jax)
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"[+] Generated {n_samples} dungeon(s) in {elapsed:.2f}s")
    
    # Build metadata
    metadata = {
        'method': 'THRML (Ising)',
        'grid_size': grid_size,
        'n_samples': n_samples,
        'generation_time': elapsed,
        'seed': seed,
        'parameters': {
            'beta': float(beta),
            'edge_bias': float(edge_bias),
            'coupling': float(coupling),
            'n_warmup': n_warmup,
            'steps_per_sample': steps_per_sample
        }
    }
    
    return dungeons, metadata


def generate_traditional(
    method: str = "cellular_automata",
    grid_size: int = 24,
    seed: int = None,
    verbose: bool = True,
    **kwargs
) -> Tuple[np.ndarray, Dict]:
    """
    Generate dungeon using traditional algorithmic methods.
    
    Args:
        method: One of ["random", "cellular_automata", "bsp", "drunkards_walk"]
        grid_size: Width and height of dungeon grid
        seed: Random seed for reproducibility
        verbose: Print generation progress
        **kwargs: Method-specific parameters (see traditional_generators.py)
        
    Returns:
        dungeon: Boolean array [grid_size, grid_size] where True = floor
        metadata: Dict with generation info
    """
    if seed is None:
        import random
        seed = random.randint(0, 999999)
    
    start_time = time.time()
    
    method = method.lower().replace(" ", "_")
    
    if verbose:
        print(f"[*] Generating {grid_size}x{grid_size} dungeon with {method}")
        print(f"    Seed: {seed}")
    
    # Map to generator functions
    generators = {
        "random": trad.random_dungeon,
        "cellular_automata": trad.cellular_automata_dungeon,
        "bsp": trad.bsp_dungeon,
        "drunkards_walk": trad.drunkards_walk_dungeon,
    }
    
    if method not in generators:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(generators.keys())}")
    
    gen_func = generators[method]
    
    # Call generator (most take width, height, seed as first args)
    if method == "random":
        dungeon = gen_func(grid_size, grid_size, kwargs.get('floor_probability', 0.5), seed)
    else:
        dungeon = gen_func(grid_size, grid_size, seed=seed, **kwargs)
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"[+] Generated dungeon in {elapsed*1000:.2f}ms")
    
    # Build metadata
    metadata = {
        'method': method,
        'grid_size': grid_size,
        'generation_time': elapsed,
        'seed': seed,
        'parameters': kwargs
    }
    
    return dungeon, metadata


def analyze_dungeon(dungeon: np.ndarray, verbose: bool = True) -> Dict:
    """
    Analyze dungeon quality and playability.
    
    Args:
        dungeon: Boolean array where True = floor
        verbose: Print analysis results
        
    Returns:
        Dict with metrics (connectivity, floor_ratio, openness, etc.)
    """
    is_connected, num_components = check_connectivity(dungeon)
    scores = calculate_playability_score(dungeon)
    
    metrics = {
        'is_connected': is_connected,
        'num_components': num_components,
        **scores
    }
    
    if verbose:
        print(f"\n[*] Dungeon Analysis:")
        print(f"    Size: {dungeon.shape[0]}x{dungeon.shape[1]}")
        print(f"    Connected: {'Yes' if is_connected else f'No ({num_components} components)'}")
        print(f"    Floor coverage: {metrics['floor_ratio']*100:.1f}%")
        print(f"    Openness: {metrics['openness']:.2f}")
        print(f"    Room quality: {metrics['room_quality']:.3f}")
    
    return metrics


if __name__ == "__main__":
    """Demo: Generate and compare dungeons"""
    import sys
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    
    # Fix Windows encoding
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("=" * 60)
    print("    EntroPit - Dungeon Generator Demo")
    print("=" * 60)
    print()
    
    seed = 42
    size = 16
    
    # Generate with THRML
    print("[1/2] THRML Generation:")
    dungeons_thrml, meta_thrml = generate_thrml(grid_size=size, n_samples=2, seed=seed, verbose=True)
    analyze_dungeon(dungeons_thrml[0], verbose=True)
    
    print()
    
    # Generate with traditional method
    print("[2/2] Traditional Generation:")
    dungeon_trad, meta_trad = generate_traditional("cellular_automata", grid_size=size, seed=seed, verbose=True)
    analyze_dungeon(dungeon_trad, verbose=True)
    
    # Visualize side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cmap = ListedColormap(['#2c3e50', '#ecf0f1'])
    
    axes[0].imshow(dungeons_thrml[0], cmap=cmap, interpolation='nearest')
    axes[0].set_title('THRML (Sample 1)')
    axes[0].axis('off')
    
    axes[1].imshow(dungeons_thrml[1], cmap=cmap, interpolation='nearest')
    axes[1].set_title('THRML (Sample 2)')
    axes[1].axis('off')
    
    axes[2].imshow(dungeon_trad, cmap=cmap, interpolation='nearest')
    axes[2].set_title('Cellular Automata')
    axes[2].axis('off')
    
    plt.tight_layout()
    import os
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/entropit_demo.png', dpi=150, bbox_inches='tight')
    print(f"\n[+] Saved comparison to 'output/entropit_demo.png'")
    plt.show()

