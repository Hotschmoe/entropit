#!/usr/bin/env python3
"""
EntroPit Quick Start Example
=============================

Minimal working example of probabilistic dungeon generation using THRML.

This script demonstrates:
- Basic THRML-based dungeon generation
- Visualization of generated dungeons
- Analysis of dungeon quality

Run:
    python examples/quickstart.py
    python examples/quickstart.py 123  # With specific seed
"""

import sys
import os
import random

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from entropit import generate_thrml, analyze_dungeon
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def visualize_dungeons(dungeons: np.ndarray, n_show: int = 4):
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
        ax.set_xticks(np.arange(-0.5, len(dungeon), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(dungeon), 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/quickstart_dungeons.png', dpi=150, bbox_inches='tight')
    print("[+] Saved visualization to 'output/quickstart_dungeons.png'")
    plt.show()


def main():
    """Main demo function"""
    # Fix Windows encoding
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("=" * 60)
    print("    EntroPit - Probabilistic Dungeon Generator")
    print("       Powered by THRML & Thermodynamic Computing")
    print("=" * 60)
    print()
    
    # Use seed from command line or generate random
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
        print(f"[*] Using seed: {seed}")
    else:
        seed = random.randint(0, 999999)
        print(f"[*] Random seed: {seed}")
        print(f"    (use 'python examples/quickstart.py {seed}' to reproduce)")
    
    print()
    
    # Generate dungeons using Gibbs sampling
    print("[1/3] Generating dungeons with THRML...")
    dungeons, metadata = generate_thrml(
        grid_size=12,
        beta=2.0,
        n_samples=4,
        seed=seed,
        verbose=True
    )
    
    print()
    
    # Analyze first dungeon
    print("[2/3] Analyzing dungeon quality...")
    metrics = analyze_dungeon(dungeons[0], verbose=True)
    
    print()
    
    # Visualize results
    print("[3/3] Creating visualization...")
    visualize_dungeons(dungeons, n_show=4)
    
    print()
    print("=" * 60)
    print("[+] Success! Your probabilistic dungeon generator works!")
    print()
    print("Experiment:")
    print("  • Run again for different dungeons (random seed each time)")
    print(f"  • Use: python examples/quickstart.py <seed> for reproducibility")
    print("  • Try different grid sizes: 8, 16, 24, 32")
    print("  • Adjust beta (temperature), edge_bias, coupling in the code")
    print()
    print("Next steps:")
    print("  • Run: python examples/comparison_demo.py")
    print("  • Run: python examples/interactive_ui.py")
    print("  • Read: docs/ARCHITECTURE.md for technical details")
    print("=" * 60)


if __name__ == "__main__":
    main()

