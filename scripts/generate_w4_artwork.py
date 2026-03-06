import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def generate_grade_lattice(output_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Grid size
    N = 4
    
    # Plot points
    for n in range(N):
        for m in range(N):
            color = 'black'
            size = 50
            if n == 0 and m == 0:
                color = 'red'
                size = 100
                ax.text(n, m - 0.2, 'VACUUM (0,0)', ha='center', va='top', fontweight='bold', color='red')
            
            ax.scatter(n, m, color=color, s=size, zorder=3)
            ax.text(n, m + 0.1, f'({n},{m})', ha='center', va='bottom', fontsize=8)

    # Draw grid lines
    for i in range(N):
        ax.plot([i, i], [0, N-1], color='gray', linestyle='--', alpha=0.5, zorder=1)
        ax.plot([0, N-1], [i, i], color='gray', linestyle='--', alpha=0.5, zorder=1)

    # Annotate towers
    ax.annotate('', xy=(3.5, 0), xytext=(0.5, 0),
                arrowprops=dict(arrowstyle="->", color='blue', lw=2))
    ax.text(2, -0.4, r'$\alpha$-tower (Op2: $n \to n+1$)', ha='center', color='blue', fontweight='bold')

    ax.annotate('', xy=(0, 3.5), xytext=(0, 0.5),
                arrowprops=dict(arrowstyle="->", color='green', lw=2))
    ax.text(-0.6, 2, r"$\beta$-tower (Op3: $m \to m+1$)", va='center', rotation=90, color='green', fontweight='bold')

    # Formatting
    ax.set_xlabel('Grade $n$', loc='right')
    ax.set_ylabel('Grade $m$', loc='top')
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xlim(-1, N)
    ax.set_ylim(-1, N)
    ax.set_title('HGST Grade Lattice $\Gamma = \mathbb{N}^2$', pad=20)
    ax.grid(False)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Generated {output_path}")

def generate_four_layer_stack(output_path):
    fig, ax = plt.subplots(figsize=(6, 8))
    
    layers = [
        "Layer 3: SU(3) (E7-SU3)",
        "Layer 2: SU(2) (E7-SU2)",
        "Layer 1: U(1) (E1)",
        "Layer 0: Grade Lattice ($\Gamma$)"
    ]
    
    colors = ['#f8d7da', '#d1ecf1', '#d4edda', '#fff3cd']
    colors.reverse() # Bottom to top
    
    height = 0.8
    spacing = 0.4
    
    for i, label in enumerate(layers[::-1]):
        y = i * (height + spacing)
        rect = patches.Rectangle((0, y), 5, height, linewidth=2, edgecolor='black', facecolor=colors[i], zorder=2)
        ax.add_patch(rect)
        ax.text(2.5, y + height/2, label, ha='center', va='center', fontweight='bold', fontsize=12)
        
        # Add internal components list for some layers
        if "Grade Lattice" in label:
            ax.text(2.5, y + 0.15, "Substrate: Axioms 1-4", ha='center', fontsize=9, style='italic')
        elif "U(1)" in label:
            ax.text(2.5, y + 0.15, "Complex Field & Gauge Invariance", ha='center', fontsize=9, style='italic')
        elif "SU(2)" in label:
            ax.text(2.5, y + 0.15, "Non-Abelian Frustration ($R>0$)", ha='center', fontsize=9, style='italic')
        elif "SU(3)" in label:
            ax.text(2.5, y + 0.15, "RegulonDB Overlap & SM Extensions", ha='center', fontsize=9, style='italic')

    # Draw vertical arrow showing inheritance/stacking
    ax.annotate('', xy=(5.5, (len(layers)-1)*(height+spacing) + height), xytext=(5.5, 0),
                arrowprops=dict(arrowstyle="->", color='black', lw=3))
    ax.text(5.7, (len(layers)/2)*(height+spacing) - 0.5, "Hierarchical Extension Stack", rotation=90, va='center', fontweight='bold')

    # Formatting
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, len(layers)*(height+spacing))
    ax.axis('off')
    ax.set_title('HGST Modular Architecture', pad=20, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Generated {output_path}")

if __name__ == "__main__":
    import os
    os.makedirs('figures', exist_ok=True)
    generate_grade_lattice('figures/grade_lattice.pdf')
    generate_four_layer_stack('figures/four_layer.pdf')
    # Also generate PNG for verification in artifacts if needed
    generate_grade_lattice('figures/grade_lattice.png')
    generate_four_layer_stack('figures/four_layer.png')
