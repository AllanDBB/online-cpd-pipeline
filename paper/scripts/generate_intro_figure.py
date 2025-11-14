"""
Script to generate comparison figure for introduction section.
Shows the same time series with high vs low noise to illustrate detection difficulty.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from synthetic_generator import generar_serie_sintetica

def create_noise_comparison_figure(save_path="paper/figures/fig_intro_noise_comparison.pdf"):
    """
    Creates a side-by-side comparison of the same time series with different noise levels.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Parameters for the base series
    longitud = 300
    num_cambios = 3
    fuerza_cambio = 4.0  # Medium-high change magnitude
    tipo_cambio = 'escalon'  # Step changes are clearer to visualize
    
    # Generate base series with LOW noise
    nivel_ruido_bajo = 0.2
    serie_bajo_ruido, puntos_cambio = generar_serie_sintetica(
        longitud=longitud,
        nivel_ruido=nivel_ruido_bajo,
        num_cambios=num_cambios,
        fuerza_cambio=fuerza_cambio,
        tipo_cambio=tipo_cambio,
        seed=42
    )
    
    # Generate the SAME series but with HIGH noise
    # We'll use the same seed and parameters but change noise
    nivel_ruido_alto = 4.5
    serie_alto_ruido, _ = generar_serie_sintetica(
        longitud=longitud,
        nivel_ruido=nivel_ruido_alto,
        num_cambios=num_cambios,
        fuerza_cambio=fuerza_cambio,
        tipo_cambio=tipo_cambio,
        seed=42  # Same seed ensures same change points
    )
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
    
    # Time axis
    t = np.arange(longitud)
    
    # LEFT PLOT: High Noise
    ax1.plot(t, serie_alto_ruido, 'b-', linewidth=1.2, alpha=0.7, label='Time series')
    for cp in puntos_cambio:
        ax1.axvline(x=cp, color='red', linestyle='--', linewidth=2, 
                   alpha=0.8, label='Change point' if cp == puntos_cambio[0] else '')
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('(a) High Noise (σ = 4.5)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)
    
    # RIGHT PLOT: Low Noise
    ax2.plot(t, serie_bajo_ruido, 'b-', linewidth=1.2, alpha=0.7, label='Time series')
    for cp in puntos_cambio:
        ax2.axvline(x=cp, color='red', linestyle='--', linewidth=2, 
                   alpha=0.8, label='Change point' if cp == puntos_cambio[0] else '')
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title('(b) Low Noise (σ = 0.2)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved to: {save_path}")
    
    # Also save as PNG for quick preview
    png_path = save_path.with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"✓ PNG preview saved to: {png_path}")
    
    plt.close()
    
    # Print statistics
    print("\n" + "="*60)
    print("FIGURE STATISTICS")
    print("="*60)
    print(f"Series length: {longitud}")
    print(f"Number of change points: {num_cambios}")
    print(f"Change points at: {puntos_cambio}")
    print(f"Change magnitude: {fuerza_cambio}")
    print(f"\nHigh noise σ: {nivel_ruido_alto}")
    print(f"High noise SNR: {fuerza_cambio / nivel_ruido_alto:.2f}")
    print(f"\nLow noise σ: {nivel_ruido_bajo}")
    print(f"Low noise SNR: {fuerza_cambio / nivel_ruido_bajo:.2f}")
    print(f"\nNoise ratio: {nivel_ruido_alto / nivel_ruido_bajo:.1f}x")
    print("="*60)

if __name__ == "__main__":
    create_noise_comparison_figure()
