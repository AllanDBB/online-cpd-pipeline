"""
Generate LOW RESOLUTION figures for faster LaTeX compilation during editing.
Run this for draft mode, then run generate_results_figures.py for final version.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configure matplotlib for DRAFT MODE (72 DPI = 4x faster)
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'figure.figsize': (10, 6),
    'figure.dpi': 72,  # LOW RESOLUTION for fast compilation
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9
})

# Paths
SYNTHETIC_CSV = r"c:\Users\allan\Documents\GitHub\online-cpd-pipeline\results\results_synthetic\10-24-2025-resultados_algoritmos_main2-synthetic.csv"
REAL_CSV = r"c:\Users\allan\Documents\GitHub\online-cpd-pipeline\results\results_real\10-24-2025-resultados_algoritmos_main3_real-real.csv"
COMPARISON_CSV = r"c:\Users\allan\Documents\GitHub\online-cpd-pipeline\results\results_comparision\10-24-2025-resultados_algoritmos_main4_comparison-comparision.csv"
OUTPUT_DIR = r"c:\Users\allan\Documents\GitHub\online-cpd-pipeline\paper\figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🚀 DRAFT MODE: Generating LOW RESOLUTION figures (72 DPI)...")
print("=" * 70)

# Import all figure generation functions from main script
import sys
sys.path.append(os.path.dirname(__file__))

# Execute main script but with low DPI already configured
exec(open('generate_results_figures.py').read())

print("\n" + "=" * 70)
print("✅ DRAFT figures generated! LaTeX will compile much faster now.")
print("⚠️  Remember to run generate_results_figures.py before final submission!")
