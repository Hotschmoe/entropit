"""
Dungeon Generator Benchmarking Framework
=========================================
Compare THRML-based generation against traditional algorithms.

Metrics:
- Generation time
- Connectivity (are all floors reachable?)
- Playability score
- Constraint satisfaction
- Visual quality metrics
"""

import time
import numpy as np
from typing import Dict, Callable, List, Tuple
import networkx as nx


def check_connectivity(dungeon: np.ndarray) -> Tuple[bool, int]:
    """
    Check if all floor tiles form a single connected component.
    
    Args:
        dungeon: Boolean array where True = floor
        
    Returns:
        (is_connected, num_components)
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
    
    Returns dict with scores:
    - floor_ratio: Percentage of floor tiles (0-1)
    - connectivity_score: 1.0 if fully connected, else 0.0
    - openness: Average number of floor neighbors per floor tile
    - room_quality: Variance in local floor density (higher = more rooms vs corridors)
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
    
    Args:
        name: Generator name
        generator_func: Function that returns a dungeon array
        n_runs: Number of runs for averaging
        
    Returns:
        Dict with benchmark results
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
    print(f"    Time: {results['avg_time_ms']:.2f}ms +/- {results['std_time_ms']:.2f}ms")
    print(f"    Connectivity: {results['connectivity_rate']*100:.0f}%")
    print(f"    Floor ratio: {results['metrics']['floor_ratio']['mean']:.2f} +/- {results['metrics']['floor_ratio']['std']:.2f}")
    print(f"    Openness: {results['metrics']['openness']['mean']:.2f} +/- {results['metrics']['openness']['std']:.2f}")
    
    return results


def compare_all_generators(grid_size: int = 24, n_runs: int = 10, seed: int = 42):
    """
    Run comprehensive benchmark of all generators.
    
    Args:
        grid_size: Size of dungeon grid
        n_runs: Number of runs per generator
        seed: Base random seed
    """
    import gen_traditional as trad
    from entropit_quickstart import create_simple_dungeon
    
    print("=" * 70)
    print("    DUNGEON GENERATOR BENCHMARK")
    print(f"    Grid Size: {grid_size}x{grid_size}")
    print(f"    Runs per generator: {n_runs}")
    print("=" * 70)
    
    generators = [
        ("Random", lambda: trad.random_dungeon(grid_size, grid_size, 0.5, seed)),
        ("Cellular Automata", lambda: trad.cellular_automata_dungeon(grid_size, grid_size, seed=seed)),
        ("BSP", lambda: trad.bsp_dungeon(grid_size, grid_size, seed=seed)),
        ("Drunkard's Walk", lambda: trad.drunkards_walk_dungeon(grid_size, grid_size, seed=seed)),
        ("THRML (Ising)", lambda: create_simple_dungeon(grid_size, seed)[0][0])
    ]
    
    results = []
    for name, gen_func in generators:
        result = benchmark_generator(name, gen_func, n_runs)
        results.append(result)
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("    COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Method':<20} {'Time (ms)':<12} {'Connected':<12} {'Floor %':<12} {'Openness':<12}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['name']:<20} "
              f"{r['avg_time_ms']:>8.1f}ms   "
              f"{r['connectivity_rate']*100:>6.0f}%      "
              f"{r['metrics']['floor_ratio']['mean']*100:>6.1f}%      "
              f"{r['metrics']['openness']['mean']:>6.2f}")
    
    print("=" * 70)
    
    # Visualize samples
    visualize_comparison(results)
    
    return results


def visualize_comparison(results: List[Dict]):
    """Create side-by-side visualization of all generators"""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    
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
    import os
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/benchmark_comparison.png', dpi=150, bbox_inches='tight')
    print("\n[+] Saved: output/benchmark_comparison.png")
    plt.show()


if __name__ == "__main__":
    import sys
    
    # Fix Windows console encoding
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    # Run benchmark
    results = compare_all_generators(grid_size=24, n_runs=10, seed=42)

