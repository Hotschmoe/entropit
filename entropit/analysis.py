"""
EntroPit - Analysis and Benchmarking
=====================================

Tools for analyzing dungeon quality and comparing generation methods.

This module provides:
- Connectivity checking (graph-based reachability analysis)
- Playability scoring (floor ratio, openness, room quality)
- Benchmarking framework for comparing generators
"""

import time
import numpy as np
from typing import Dict, Callable, List, Tuple, Optional
import networkx as nx


def check_connectivity(dungeon: np.ndarray) -> Tuple[bool, int]:
    """
    Check if all floor tiles form a single connected component.
    
    Uses NetworkX to build a graph of floor tiles and count connected components.
    A dungeon is "playable" if all floors are reachable from any starting point.
    
    Args:
        dungeon: Boolean array where True = floor
        
    Returns:
        (is_connected, num_components) where:
            - is_connected: True if all floors form one connected region
            - num_components: Number of disconnected floor regions
            
    Example:
        >>> dungeon = generate_thrml(grid_size=16)[0][0]
        >>> connected, components = check_connectivity(dungeon)
        >>> if connected:
        ...     print("Dungeon is fully connected!")
    """
    height, width = dungeon.shape
    
    # Build graph of floor tiles
    G = nx.Graph()
    
    # Add all floor tiles as nodes
    floor_tiles = []
    for i in range(height):
        for j in range(width):
            if dungeon[i, j]:
                floor_tiles.append((i, j))
                G.add_node((i, j))
    
    if len(floor_tiles) == 0:
        return False, 0
    
    # Add edges between adjacent floor tiles (4-connected)
    for i, j in floor_tiles:
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < height and 0 <= nj < width and dungeon[ni, nj]:
                G.add_edge((i, j), (ni, nj))
    
    # Count connected components
    num_components = nx.number_connected_components(G)
    is_connected = num_components == 1
    
    return is_connected, num_components


def calculate_playability_score(dungeon: np.ndarray) -> Dict[str, float]:
    """
    Heuristic playability metrics.
    
    Analyzes dungeon characteristics that affect gameplay quality:
    - Floor ratio: How much of the dungeon is explorable
    - Connectivity: Whether all areas are reachable
    - Openness: How cramped vs spacious the dungeon feels
    - Room quality: Distinction between rooms and corridors
    
    Returns dict with scores:
        - floor_ratio: Percentage of floor tiles (0-1)
        - connectivity_score: 1.0 if fully connected, else 0.0
        - openness: Average number of floor neighbors per floor tile (0-1)
        - room_quality: Variance in local floor density (higher = more rooms vs corridors)
        
    Example:
        >>> scores = calculate_playability_score(dungeon)
        >>> print(f"Floor coverage: {scores['floor_ratio']*100:.1f}%")
        >>> print(f"Openness: {scores['openness']:.2f}")
    """
    height, width = dungeon.shape
    total_tiles = height * width
    floor_tiles = np.sum(dungeon)
    
    if floor_tiles == 0:
        return {
            'floor_ratio': 0.0,
            'connectivity_score': 0.0,
            'openness': 0.0,
            'room_quality': 0.0
        }
    
    # Floor ratio
    floor_ratio = floor_tiles / total_tiles
    
    # Connectivity
    is_connected, _ = check_connectivity(dungeon)
    connectivity_score = 1.0 if is_connected else 0.0
    
    # Openness: average neighbors per floor tile
    neighbor_counts = []
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if dungeon[i, j]:
                # Count 4-connected floor neighbors
                neighbors = sum([
                    dungeon[i-1, j], dungeon[i+1, j],
                    dungeon[i, j-1], dungeon[i, j+1]
                ])
                neighbor_counts.append(neighbors)
    
    openness = np.mean(neighbor_counts) if neighbor_counts else 0.0
    
    # Room quality: local density variance
    # Higher variance = more distinction between open rooms and narrow corridors
    local_densities = []
    for i in range(2, height - 2):
        for j in range(2, width - 2):
            if dungeon[i, j]:
                # 5x5 window density
                window = dungeon[i-2:i+3, j-2:j+3]
                local_densities.append(np.mean(window))
    
    room_quality = np.var(local_densities) if local_densities else 0.0
    
    return {
        'floor_ratio': float(floor_ratio),
        'connectivity_score': float(connectivity_score),
        'openness': float(openness / 4.0),  # Normalize to 0-1
        'room_quality': float(room_quality)
    }


def benchmark_generator(
    name: str,
    generator_func: Callable[[], np.ndarray],
    n_runs: int = 10
) -> Dict:
    """
    Benchmark a dungeon generator.
    
    Runs the generator multiple times and computes aggregate statistics
    for generation time, connectivity rate, and quality metrics.
    
    Args:
        name: Generator name (for display)
        generator_func: Function that returns a dungeon array (no arguments)
        n_runs: Number of runs for averaging
        
    Returns:
        Dict with benchmark results:
            - name: Generator name
            - n_runs: Number of iterations
            - avg_time_ms: Average generation time in milliseconds
            - std_time_ms: Standard deviation of generation time
            - connectivity_rate: Fraction of generated dungeons that are connected
            - metrics: Dict of averaged playability metrics
            - sample_dungeon: First generated dungeon (for visualization)
            
    Example:
        >>> from entropit import generate_thrml
        >>> results = benchmark_generator(
        ...     "THRML",
        ...     lambda: generate_thrml(grid_size=24, verbose=False)[0][0],
        ...     n_runs=10
        ... )
        >>> print(f"{results['name']}: {results['avg_time_ms']:.1f}ms")
    """
    print(f"\n[*] Benchmarking: {name}")
    print(f"    Running {n_runs} iterations...")
    
    times = []
    all_scores = {
        'floor_ratio': [],
        'connectivity_score': [],
        'openness': [],
        'room_quality': []
    }
    
    successful_connections = 0
    
    for i in range(n_runs):
        # Time generation
        start = time.time()
        dungeon = generator_func()
        elapsed = time.time() - start
        times.append(elapsed)
        
        # Calculate metrics
        scores = calculate_playability_score(dungeon)
        
        for key in all_scores:
            all_scores[key].append(scores[key])
        
        if scores['connectivity_score'] > 0.5:
            successful_connections += 1
        
        if i == 0:
            # Save first sample for visualization
            first_sample = dungeon
    
    # Aggregate results
    results = {
        'name': name,
        'n_runs': n_runs,
        'avg_time_ms': np.mean(times) * 1000,
        'std_time_ms': np.std(times) * 1000,
        'connectivity_rate': successful_connections / n_runs,
        'metrics': {
            key: {
                'mean': np.mean(values),
                'std': np.std(values)
            }
            for key, values in all_scores.items()
        },
        'sample_dungeon': first_sample
    }
    
    # Print summary
    print(f"    Time: {results['avg_time_ms']:.2f}ms ± {results['std_time_ms']:.2f}ms")
    print(f"    Connectivity: {results['connectivity_rate']*100:.0f}%")
    print(f"    Floor ratio: {results['metrics']['floor_ratio']['mean']:.2f} ± {results['metrics']['floor_ratio']['std']:.2f}")
    print(f"    Openness: {results['metrics']['openness']['mean']:.2f} ± {results['metrics']['openness']['std']:.2f}")
    
    return results


def visualize_comparison(results: List[Dict], output_path: str = "output/benchmark_comparison.png"):
    """
    Create side-by-side visualization of all generators.
    
    Args:
        results: List of benchmark result dicts from benchmark_generator()
        output_path: Where to save the comparison image
        
    Example:
        >>> results = [benchmark_generator(...) for generator in generators]
        >>> visualize_comparison(results)
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import os
    
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    
    if n == 1:
        axes = [axes]
    
    cmap = ListedColormap(['#2c3e50', '#ecf0f1'])
    
    for ax, result in zip(axes, results):
        dungeon = result['sample_dungeon']
        ax.imshow(dungeon, cmap=cmap, interpolation='nearest')
        
        # Title with key metrics
        connectivity = "✓" if result['connectivity_rate'] > 0.8 else "✗"
        title = (f"{result['name']}\n"
                f"{connectivity} {result['avg_time_ms']:.1f}ms | "
                f"{result['metrics']['floor_ratio']['mean']*100:.0f}% floor")
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[+] Saved: {output_path}")
    plt.close()

