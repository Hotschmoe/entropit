"""
EntroPit - Interactive User Interface
======================================

Gradio-based web interface for interactive dungeon generation and comparison.

Features:
- Generate dungeons with different algorithms
- Adjust THRML parameters in real-time
- View connectivity and quality metrics
- Side-by-side comparisons
"""

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import io
from PIL import Image
from typing import Tuple

from .core import generate_thrml, analyze_dungeon
from .traditional import generate_traditional
from .analysis import check_connectivity, calculate_playability_score


def render_dungeon(dungeon: np.ndarray, title: str = "") -> Image.Image:
    """
    Convert dungeon array to PIL Image for display in Gradio.
    
    Args:
        dungeon: Boolean array where True = floor
        title: Title to display above the dungeon
        
    Returns:
        PIL Image object
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = ListedColormap(['#2c3e50', '#ecf0f1'])
    
    ax.imshow(dungeon, cmap=cmap, interpolation='nearest')
    ax.set_title(title, fontsize=14, pad=10)
    ax.axis('off')
    
    # Add grid
    ax.set_xticks(np.arange(-0.5, len(dungeon), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(dungeon), 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    
    return img


def generate_traditional_ui(method: str, size: int, seed: int) -> Tuple[Image.Image, str]:
    """
    Gradio handler for traditional method generation.
    
    Args:
        method: Algorithm name (display format)
        size: Grid size
        seed: Random seed
        
    Returns:
        (image, metrics_markdown)
    """
    method_map = {
        "Random": "random",
        "Cellular Automata": "cellular_automata",
        "BSP": "bsp",
        "Drunkard's Walk": "drunkards_walk"
    }
    
    dungeon, metadata = generate_traditional(method_map[method], grid_size=size, seed=seed, verbose=False)
    
    # Calculate metrics
    is_connected, num_components = check_connectivity(dungeon)
    scores = calculate_playability_score(dungeon)
    
    # Create visualization
    img = render_dungeon(dungeon, f"{method} Dungeon")
    
    # Format metrics
    metrics_text = f"""
### Metrics:
- **Connected**: {'✅ Yes' if is_connected else f'❌ No ({num_components} components)'}
- **Floor Coverage**: {scores['floor_ratio']*100:.1f}%
- **Openness**: {scores['openness']:.2f}
- **Room Quality**: {scores['room_quality']:.3f}
- **Generation Time**: {metadata['generation_time']*1000:.2f}ms
"""
    
    return img, metrics_text


def generate_thrml_ui(size: int, beta: float, edge_bias: float, coupling: float, seed: int) -> Tuple[Image.Image, str]:
    """
    Gradio handler for THRML generation.
    
    Args:
        size: Grid size
        beta: Inverse temperature
        edge_bias: Edge tile bias
        coupling: Neighbor coupling strength
        seed: Random seed
        
    Returns:
        (image, metrics_markdown)
    """
    dungeons, metadata = generate_thrml(
        grid_size=size,
        beta=beta,
        edge_bias=edge_bias,
        coupling=coupling,
        n_samples=1,
        seed=seed,
        verbose=False
    )
    dungeon = dungeons[0]
    
    # Calculate metrics
    is_connected, num_components = check_connectivity(dungeon)
    scores = calculate_playability_score(dungeon)
    
    # Create visualization
    img = render_dungeon(dungeon, f"THRML Dungeon (β={beta})")
    
    # Format metrics
    metrics_text = f"""
### Metrics:
- **Connected**: {'✅ Yes' if is_connected else f'❌ No ({num_components} components)'}
- **Floor Coverage**: {scores['floor_ratio']*100:.1f}%
- **Openness**: {scores['openness']:.2f}
- **Room Quality**: {scores['room_quality']:.3f}
- **Generation Time**: {metadata['generation_time']:.2f}s

### Parameters:
- Temperature (β): {beta}
- Edge Bias: {edge_bias}
- Coupling Strength: {coupling}
"""
    
    return img, metrics_text


def compare_all(size: int, seed: int) -> Image.Image:
    """
    Generate with all methods for side-by-side comparison.
    
    Args:
        size: Grid size
        seed: Random seed
        
    Returns:
        PIL Image with all methods displayed
    """
    method_names = ["Random", "Cellular Automata", "BSP", "Drunkard's Walk"]
    method_keys = ["random", "cellular_automata", "bsp", "drunkards_walk"]
    
    # Create subplot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    cmap = ListedColormap(['#2c3e50', '#ecf0f1'])
    
    # Traditional methods
    for idx, (name, key) in enumerate(zip(method_names, method_keys)):
        dungeon, _ = generate_traditional(key, grid_size=size, seed=seed, verbose=False)
        
        is_connected, _ = check_connectivity(dungeon)
        scores = calculate_playability_score(dungeon)
        
        axes[idx].imshow(dungeon, cmap=cmap, interpolation='nearest')
        conn_icon = "✓" if is_connected else "✗"
        axes[idx].set_title(f"{name}\n{conn_icon} {scores['floor_ratio']*100:.0f}% floor", 
                           fontsize=10)
        axes[idx].axis('off')
    
    # THRML
    dungeons, _ = generate_thrml(grid_size=size, n_samples=1, seed=seed, verbose=False)
    dungeon = dungeons[0]
    is_connected, _ = check_connectivity(dungeon)
    scores = calculate_playability_score(dungeon)
    
    axes[4].imshow(dungeon, cmap=cmap, interpolation='nearest')
    conn_icon = "✓" if is_connected else "✗"
    axes[4].set_title(f"THRML (Ising)\n{conn_icon} {scores['floor_ratio']*100:.0f}% floor", 
                     fontsize=10)
    axes[4].axis('off')
    
    # Hide last subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    
    return img


def create_ui():
    """
    Create and return the Gradio interface.
    
    Returns:
        gr.Blocks: Gradio interface object
    """
    with gr.Blocks(title="EntroPit - Probabilistic Dungeon Generator") as demo:
        gr.Markdown("""
        # EntroPit - Probabilistic Dungeon Generator
        
        Compare **THRML** (thermodynamic computing) against traditional procedural algorithms.
        
        **Key Insight**: THRML naturally handles constraints (like connectivity) that traditional 
        methods struggle with - at the cost of computation time (which Extropic hardware will solve!).
        """)
        
        with gr.Tabs():
            # Tab 1: Traditional Methods
            with gr.Tab("Traditional Methods"):
                gr.Markdown("### Classic Procedural Algorithms")
                
                with gr.Row():
                    with gr.Column():
                        trad_method = gr.Dropdown(
                            choices=["Random", "Cellular Automata", "BSP", "Drunkard's Walk"],
                            value="Cellular Automata",
                            label="Algorithm"
                        )
                        trad_size = gr.Slider(8, 32, value=24, step=4, label="Grid Size")
                        trad_seed = gr.Number(value=42, label="Seed", precision=0)
                        trad_button = gr.Button("Generate", variant="primary")
                    
                    with gr.Column():
                        trad_output = gr.Image(label="Generated Dungeon")
                        trad_metrics = gr.Markdown()
                
                trad_button.click(
                    fn=generate_traditional_ui,
                    inputs=[trad_method, trad_size, trad_seed],
                    outputs=[trad_output, trad_metrics]
                )
            
            # Tab 2: THRML
            with gr.Tab("THRML (Probabilistic)"):
                gr.Markdown("""
                ### Thermodynamic Computing Approach
                Uses Gibbs sampling on an Ising model to generate dungeons by minimizing energy.
                """)
                
                with gr.Row():
                    with gr.Column():
                        thrml_size = gr.Slider(8, 24, value=12, step=4, label="Grid Size")
                        thrml_beta = gr.Slider(0.5, 5.0, value=2.0, step=0.5, 
                                              label="Temperature (β) - Higher = More Structure")
                        thrml_edge_bias = gr.Slider(-5.0, 0.0, value=-2.0, step=0.5,
                                                   label="Edge Bias - More Negative = Thicker Walls")
                        thrml_coupling = gr.Slider(0.3, 2.0, value=0.8, step=0.1,
                                                  label="Coupling Strength - Higher = Bigger Rooms")
                        thrml_seed = gr.Number(value=42, label="Seed", precision=0)
                        thrml_button = gr.Button("Generate (Slower - Uses Gibbs Sampling)", variant="primary")
                    
                    with gr.Column():
                        thrml_output = gr.Image(label="Generated Dungeon")
                        thrml_metrics = gr.Markdown()
                
                thrml_button.click(
                    fn=generate_thrml_ui,
                    inputs=[thrml_size, thrml_beta, thrml_edge_bias, thrml_coupling, thrml_seed],
                    outputs=[thrml_output, thrml_metrics]
                )
            
            # Tab 3: Compare All
            with gr.Tab("Compare All"):
                gr.Markdown("""
                ### Side-by-Side Comparison
                Generate with all methods simultaneously to see the differences.
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        compare_size = gr.Slider(8, 24, value=16, step=4, label="Grid Size")
                        compare_seed = gr.Number(value=42, label="Seed", precision=0)
                        compare_button = gr.Button("Generate All", variant="primary")
                    
                    with gr.Column(scale=3):
                        compare_output = gr.Image(label="Comparison")
                
                compare_button.click(
                    fn=compare_all,
                    inputs=[compare_size, compare_seed],
                    outputs=compare_output
                )
        
        gr.Markdown("""
        ---
        ### Learn More
        - **README.md** - Project overview and getting started
        - **docs/ARCHITECTURE.md** - Mathematical formulation
        - **examples/benchmark_demo.py** - Run full performance comparison
        
        ### Key Takeaways
        1. **Traditional methods are fast** but struggle with global constraints (connectivity)
        2. **THRML is slower** but naturally enforces constraints through energy minimization
        3. **Extropic hardware** will make THRML orders of magnitude faster, enabling real-time generation
        """)
    
    return demo


def launch_ui(share: bool = False):
    """
    Launch the Gradio UI.
    
    Args:
        share: Whether to create a public share link
    """
    import sys
    
    # Fix Windows encoding
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    demo = create_ui()
    demo.launch(share=share)


if __name__ == "__main__":
    launch_ui()