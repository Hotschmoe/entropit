"""
Traditional Dungeon Generators
===============================
Baseline algorithms for comparison with THRML-based generation.

Implementations:
- Random (baseline)
- Cellular Automata (Game of Life-style)
- BSP (Binary Space Partitioning)
- Drunkard's Walk

These serve as benchmarks to demonstrate THRML's advantages.
"""

import numpy as np
from typing import Tuple
import random


def random_dungeon(width: int, height: int, floor_probability: float = 0.5, seed: int = None) -> np.ndarray:
    """
    Baseline: purely random floor/wall placement.
    
    Args:
        width: Grid width
        height: Grid height
        floor_probability: Probability each tile is a floor (0-1)
        seed: Random seed
        
    Returns:
        Boolean array [height, width] where True = floor
    """
    if seed is not None:
        np.random.seed(seed)
    
    dungeon = np.random.random((height, width)) < floor_probability
    
    # Force edges to be walls
    dungeon[0, :] = False
    dungeon[-1, :] = False
    dungeon[:, 0] = False
    dungeon[:, -1] = False
    
    return dungeon


def cellular_automata_dungeon(width: int, height: int, 
                               initial_floor_prob: float = 0.45,
                               iterations: int = 5,
                               birth_limit: int = 4,
                               death_limit: int = 3,
                               seed: int = None) -> np.ndarray:
    """
    Cellular automata (similar to Conway's Game of Life).
    
    Classic roguelike technique - creates organic cave-like structures.
    
    Args:
        width: Grid width
        height: Grid height
        initial_floor_prob: Initial random floor density
        iterations: Number of CA steps
        birth_limit: Neighbors needed for wall->floor
        death_limit: Neighbors needed to stay floor
        seed: Random seed
        
    Returns:
        Boolean array [height, width] where True = floor
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize randomly
    dungeon = np.random.random((height, width)) < initial_floor_prob
    
    # Force edges to be walls
    dungeon[0, :] = False
    dungeon[-1, :] = False
    dungeon[:, 0] = False
    dungeon[:, -1] = False
    
    # Run cellular automata iterations
    for _ in range(iterations):
        new_dungeon = dungeon.copy()
        
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                # Count floor neighbors (8-connected)
                neighbors = np.sum(dungeon[i-1:i+2, j-1:j+2]) - dungeon[i, j]
                
                if dungeon[i, j]:  # Currently floor
                    # Stay floor if enough neighbors are floor
                    new_dungeon[i, j] = neighbors >= death_limit
                else:  # Currently wall
                    # Become floor if enough neighbors are floor
                    new_dungeon[i, j] = neighbors >= birth_limit
        
        dungeon = new_dungeon
    
    return dungeon


def bsp_dungeon(width: int, height: int, 
                min_room_size: int = 4,
                max_room_size: int = 10,
                seed: int = None) -> np.ndarray:
    """
    Binary Space Partitioning with rooms and corridors.
    
    Classic algorithm used in Nethack, ADOM, etc.
    Guaranteed connectivity but very rigid/geometric.
    
    Args:
        width: Grid width
        height: Grid height
        min_room_size: Minimum room dimension
        max_room_size: Maximum room dimension
        seed: Random seed
        
    Returns:
        Boolean array [height, width] where True = floor
    """
    if seed is not None:
        random.seed(seed)
    
    dungeon = np.zeros((height, width), dtype=bool)
    
    # Simple BSP implementation
    def split_region(x, y, w, h, depth=0):
        """Recursively split regions and add rooms"""
        if depth > 3 or w < min_room_size * 2 or h < min_room_size * 2:
            # Create a room in this region
            # Ensure room fits with bounds checking
            max_room_w = min(max_room_size, w - 2)
            max_room_h = min(max_room_size, h - 2)
            
            if max_room_w < min_room_size or max_room_h < min_room_size:
                # Region too small, skip
                return None
            
            room_w = random.randint(min_room_size, max_room_w)
            room_h = random.randint(min_room_size, max_room_h)
            
            # Position with bounds checking
            max_x_offset = max(0, w - room_w - 1)
            max_y_offset = max(0, h - room_h - 1)
            
            if max_x_offset == 0 or max_y_offset == 0:
                return None
                
            room_x = x + random.randint(1, max_x_offset)
            room_y = y + random.randint(1, max_y_offset)
            
            # Carve room
            dungeon[room_y:room_y + room_h, room_x:room_x + room_w] = True
            return (room_x + room_w // 2, room_y + room_h // 2)  # Center point
        
        # Split either horizontally or vertically
        if random.random() < 0.5:  # Vertical split
            split_x = x + random.randint(min_room_size, w - min_room_size)
            center1 = split_region(x, y, split_x - x, h, depth + 1)
            center2 = split_region(split_x, y, w - (split_x - x), h, depth + 1)
            
            # Connect with horizontal corridor
            if center1 and center2:
                y_corridor = random.randint(y + 1, y + h - 2)
                for cx in range(min(center1[0], center2[0]), max(center1[0], center2[0]) + 1):
                    dungeon[y_corridor, cx] = True
        else:  # Horizontal split
            split_y = y + random.randint(min_room_size, h - min_room_size)
            center1 = split_region(x, y, w, split_y - y, depth + 1)
            center2 = split_region(x, split_y, w, h - (split_y - y), depth + 1)
            
            # Connect with vertical corridor
            if center1 and center2:
                x_corridor = random.randint(x + 1, x + w - 2)
                for cy in range(min(center1[1], center2[1]), max(center1[1], center2[1]) + 1):
                    dungeon[cy, x_corridor] = True
        
        return center1 if center1 else center2
    
    split_region(0, 0, width, height)
    
    return dungeon


def drunkards_walk_dungeon(width: int, height: int,
                            floor_percentage: float = 0.4,
                            seed: int = None) -> np.ndarray:
    """
    Drunkard's walk algorithm.
    
    Start at center, randomly walk and carve floors until target percentage reached.
    Creates meandering organic corridors.
    
    Args:
        width: Grid width
        height: Grid height
        floor_percentage: Target floor coverage (0-1)
        seed: Random seed
        
    Returns:
        Boolean array [height, width] where True = floor
    """
    if seed is not None:
        random.seed(seed)
    
    dungeon = np.zeros((height, width), dtype=bool)
    
    # Start at center
    x, y = width // 2, height // 2
    
    target_floors = int(width * height * floor_percentage)
    current_floors = 0
    
    # Random walk
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    while current_floors < target_floors:
        # Carve current position
        if not dungeon[y, x]:
            dungeon[y, x] = True
            current_floors += 1
        
        # Random step
        dx, dy = random.choice(directions)
        new_x = max(1, min(width - 2, x + dx))
        new_y = max(1, min(height - 2, y + dy))
        
        x, y = new_x, new_y
    
    return dungeon


if __name__ == "__main__":
    """Quick test of all generators"""
    import sys
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    
    # Fix Windows console encoding
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    seed = 42
    size = 24
    
    generators = [
        ("Random", lambda: random_dungeon(size, size, 0.5, seed)),
        ("Cellular Automata", lambda: cellular_automata_dungeon(size, size, seed=seed)),
        ("BSP", lambda: bsp_dungeon(size, size, seed=seed)),
        ("Drunkard's Walk", lambda: drunkards_walk_dungeon(size, size, seed=seed))
    ]
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    cmap = ListedColormap(['#2c3e50', '#ecf0f1'])
    
    for ax, (name, gen_func) in zip(axes, generators):
        dungeon = gen_func()
        ax.imshow(dungeon, cmap=cmap, interpolation='nearest')
        ax.set_title(f'{name}\n{np.sum(dungeon)} floors ({100*np.mean(dungeon):.1f}%)')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('traditional_comparison.png', dpi=150, bbox_inches='tight')
    print("[+] Generated traditional_comparison.png")
    plt.show()

