#!/usr/bin/env python3
"""
EntroPit Comparison Demo
========================

Side-by-side comparison of THRML vs traditional dungeon generation methods.

This demonstrates:
- All 5 generation methods (Random, CA, BSP, Drunkard's Walk, THRML)
- Quality metrics for each method
- Visual comparison
- Performance benchmarking

Run:
    python examples/comparison_demo.py
"""

import sys
import os

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from entropit import generate_thrml, generate_traditional, analyze_dungeon
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def main():
    """Compare all generation methods"""
    # Fix Windows encoding
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("=" * 70)
    print("    EntroPit - Method Comparison Demo")
    print("=" * 70)
    print()
    
    seed = 42
    size = 24
    
    # Define all methods
    methods = [
        ("Random", lambda: generate_traditional("random", grid_size=size, seed=seed, verbose=False)),
        ("Cellular Automata", lambda: generate_traditional("cellular_automata", grid_size=size, seed=seed, verbose=False)),
        ("BSP", lambda: generate_traditional("bsp", grid_size=size, seed=seed, verbose=False)),
        ("Drunkard's Walk", lambda: generate_traditional("drunkards_walk", grid_size=size, seed=seed, verbose=False)),
        ("THRML (Ising)", lambda: (generate_thrml(grid_size=size, seed=seed, verbose=False)[0][0], {'method': 'THRML'}))
    ]
    
    dungeons = []
    metadata_list = []
    
    print(f"[*] Generating {len(methods)} dungeons (seed={seed}, size={size}x{size})\n")
    
    # Generate all dungeons
    for name, gen_func in methods:
        print(f"[*] Generating: {name}...")
        result = gen_func()
        if isinstance(result, tuple):
            dungeon, metadata = result
        else:
            dungeon = result
            metadata = {'method': name}
        
        dungeons.append(dungeon)
        metadata_list.append(metadata)
        
        # Analyze
        metrics = analyze_dungeon(dungeon, verbose=False)
        
        gen_time = metadata.get('generation_time', 0)
        conn_status = "[✓] Connected" if metrics['is_connected'] else f"[✗] Disconnected ({metrics['num_components']} components)"
        
        print(f"    {conn_status}")
        print(f"    Floor: {metrics['floor_ratio']*100:.1f}% | Openness: {metrics['openness']:.2f} | Time: {gen_time*1000:.2f}ms")
        print()
    
    # Create comparison visualization
    print("[*] Creating comparison visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    cmap = ListedColormap(['#2c3e50', '#ecf0f1'])
    
    for idx, (dungeon, (name, _)) in enumerate(zip(dungeons, methods)):
        if idx >= len(axes):
            break
            
        from entropit.analysis import check_connectivity, calculate_playability_score
        
        is_connected, _ = check_connectivity(dungeon)
        scores = calculate_playability_score(dungeon)
        
        axes[idx].imshow(dungeon, cmap=cmap, interpolation='nearest')
        conn_icon = "[✓]" if is_connected else "[✗]"
        axes[idx].set_title(
            f"{name}\n{conn_icon} {scores['floor_ratio']*100:.0f}% floor | Openness: {scores['openness']:.2f}",
            fontsize=10
        )
        axes[idx].axis('off')
    
    # Hide unused subplot
    if len(methods) < len(axes):
        axes[-1].axis('off')
    
    plt.suptitle("EntroPit Method Comparison", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/comparison_demo.png', dpi=150, bbox_inches='tight')
    print("[+] Saved: output/comparison_demo.png")
    plt.show()
    
    print()
    print("=" * 70)
    print("[+] Comparison complete!")
    print()
    print("Key observations:")
    print("  • Traditional methods are FAST (< 1ms) but often disconnected")
    print("  • BSP guarantees connectivity through explicit corridors")
    print("  • THRML is slower but achieves connectivity through energy minimization")
    print("  • Each method produces distinctive patterns")
    print()
    print("Next steps:")
    print("  • Run: python examples/benchmark_demo.py for detailed statistics")
    print("  • Run: python examples/interactive_ui.py for live parameter tuning")
    print("=" * 70)


if __name__ == "__main__":
    main()

