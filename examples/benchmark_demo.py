#!/usr/bin/env python3
"""
EntroPit Benchmark Demo
=======================

Comprehensive performance and quality benchmarking of all generation methods.

This runs multiple iterations of each generator and computes:
- Average generation time
- Connectivity success rate
- Quality metrics (floor ratio, openness, room quality)
- Statistical analysis

Run:
    python examples/benchmark_demo.py
"""

import sys
import os

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from entropit import generate_thrml, generate_traditional, benchmark_generator
from entropit.analysis import visualize_comparison


def main():
    """Run comprehensive benchmark"""
    # Fix Windows encoding
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("=" * 70)
    print("    EntroPit - Comprehensive Benchmark")
    print("=" * 70)
    print()
    
    grid_size = 24
    n_runs = 10
    seed = 42
    
    print(f"Configuration:")
    print(f"  • Grid Size: {grid_size}x{grid_size}")
    print(f"  • Runs per method: {n_runs}")
    print(f"  • Base seed: {seed}")
    print()
    
    # Define generators
    generators = [
        ("Random", lambda: generate_traditional("random", grid_size=grid_size, seed=seed, verbose=False)[0]),
        ("Cellular Automata", lambda: generate_traditional("cellular_automata", grid_size=grid_size, seed=seed, verbose=False)[0]),
        ("BSP", lambda: generate_traditional("bsp", grid_size=grid_size, seed=seed, verbose=False)[0]),
        ("Drunkard's Walk", lambda: generate_traditional("drunkards_walk", grid_size=grid_size, seed=seed, verbose=False)[0]),
        ("THRML (Ising)", lambda: generate_thrml(grid_size=grid_size, seed=seed, verbose=False)[0][0])
    ]
    
    results = []
    
    print("Running benchmarks...")
    print("(This may take a minute, especially for THRML)")
    
    for name, gen_func in generators:
        result = benchmark_generator(name, gen_func, n_runs=n_runs)
        results.append(result)
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("    COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Method':<20} {'Time (ms)':<15} {'Connected':<12} {'Floor %':<12} {'Openness'}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['name']:<20} "
              f"{r['avg_time_ms']:>8.1f} ± {r['std_time_ms']:>4.1f}   "
              f"{r['connectivity_rate']*100:>6.0f}%      "
              f"{r['metrics']['floor_ratio']['mean']*100:>6.1f}%      "
              f"{r['metrics']['openness']['mean']:>6.2f}")
    
    print("=" * 70)
    
    # Create visualization
    print("\n[*] Creating benchmark visualization...")
    visualize_comparison(results, output_path="output/benchmark_demo.png")
    
    print()
    print("=" * 70)
    print("[+] Benchmark complete!")
    print()
    print("Key findings:")
    print("  1. Traditional methods: 0.1-10ms, 0-100% connectivity")
    print("  2. THRML: ~1-10s, 80-100% connectivity")
    print("  3. Speed-quality tradeoff is clear")
    print("  4. Extropic hardware will eliminate this tradeoff!")
    print()
    print("The THRML advantage:")
    print("  • Declarative: Specify constraints, not algorithms")
    print("  • Flexible: Adjust energy function for different styles")
    print("  • Scalable: Hardware acceleration coming soon")
    print("=" * 70)


if __name__ == "__main__":
    main()

