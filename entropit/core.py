"""
EntroPit - Core THRML-based Generation
=======================================

THRML-powered dungeon generation using Ising energy-based models.
This module provides the main API for probabilistic dungeon generation.
"""

import numpy as np
import jax.numpy as jnp
from typing import Tuple, Dict, Optional
import time

# Import THRML components
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init
import jax

from .analysis import check_connectivity, calculate_playability_score


def generate_thrml(
    grid_size: int = 12,
    beta: float = 2.0,
    edge_bias: float = -2.0,
    coupling: float = 0.8,
    n_warmup: int = 200,
    n_samples: int = 1,
    steps_per_sample: int = 20,
    seed: Optional[int] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Generate dungeon(s) using THRML Ising model.
    
    This function creates dungeons by modeling each tile as a node in an
    Ising-like energy-based model, then using Gibbs sampling to find
    low-energy (high-quality) configurations.
    
    Args:
        grid_size: Width and height of dungeon grid
        beta: Inverse temperature (higher = more structured, range: 0.5-5.0)
        edge_bias: Bias for edge tiles (negative = prefer walls, range: -5.0 to 0.0)
        coupling: Neighbor interaction strength (higher = bigger rooms, range: 0.3-2.0)
        n_warmup: Equilibration steps before sampling
        n_samples: Number of dungeon variants to generate
        steps_per_sample: Steps between samples (for decorrelation)
        seed: Random seed for reproducibility (None = random)
        verbose: Print generation progress
        
    Returns:
        dungeons: Array [n_samples, grid_size, grid_size] of boolean arrays (True = floor)
        metadata: Dict with generation info (time, parameters, connectivity, etc.)
        
    Example:
        >>> dungeons, meta = generate_thrml(grid_size=16, beta=2.0, seed=42)
        >>> print(f"Generated {len(dungeons)} dungeons in {meta['generation_time']:.2f}s")
        >>> print(f"Connectivity: {meta['connectivity_rate']*100:.0f}%")
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
    
    # Analyze connectivity
    connectivity_count = sum(1 for d in dungeons if check_connectivity(d)[0])
    connectivity_rate = connectivity_count / n_samples
    
    if verbose:
        print(f"[+] Generated {n_samples} dungeon(s) in {elapsed:.2f}s")
        print(f"    Connectivity: {connectivity_rate*100:.0f}% ({connectivity_count}/{n_samples})")
    
    # Build metadata
    metadata = {
        'method': 'THRML (Ising)',
        'grid_size': grid_size,
        'n_samples': n_samples,
        'generation_time': elapsed,
        'seed': seed,
        'connectivity_rate': connectivity_rate,
        'parameters': {
            'beta': float(beta),
            'edge_bias': float(edge_bias),
            'coupling': float(coupling),
            'n_warmup': n_warmup,
            'steps_per_sample': steps_per_sample
        }
    }
    
    return dungeons, metadata


def analyze_dungeon(dungeon: np.ndarray, verbose: bool = True) -> Dict:
    """
    Analyze dungeon quality and playability.
    
    Computes various metrics including connectivity, floor coverage,
    openness, and room quality.
    
    Args:
        dungeon: Boolean array where True = floor
        verbose: Print analysis results to stdout
        
    Returns:
        Dict with metrics:
            - is_connected: bool (are all floors reachable?)
            - num_components: int (number of disconnected regions)
            - floor_ratio: float (percentage of floor tiles)
            - openness: float (average neighbors per tile, 0-1)
            - room_quality: float (variance in local density)
            
    Example:
        >>> dungeon = dungeons[0]
        >>> metrics = analyze_dungeon(dungeon, verbose=True)
        >>> if metrics['is_connected']:
        ...     print("Dungeon is fully connected!")
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

