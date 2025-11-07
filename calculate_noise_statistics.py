"""
Calculate noise statistics from crime time series classification data.
Used to demonstrate high noise levels in real crime data for the paper.
"""
import pandas as pd
import numpy as np

# Load classification data
csv_path = r"c:\Users\allan\Documents\GitHub\online-cpd-pipeline\results\results_real\10-31-2025-clasificacion_series_criminalidad.csv"
df = pd.read_csv(csv_path)

print("=" * 80)
print("NOISE STATISTICS FOR CRIME TIME SERIES DATASET")
print("=" * 80)

# Overall statistics
print("\n📊 OVERALL NOISE STATISTICS (NSR - Noise-to-Signal Ratio):")
print(f"  Total series: {len(df)}")
print(f"  Min NSR: {df['ruido'].min():.4f}")
print(f"  Max NSR: {df['ruido'].max():.4f}")
print(f"  Mean NSR: {df['ruido'].mean():.4f}")
print(f"  Median NSR: {df['ruido'].median():.4f}")
print(f"  Std Dev: {df['ruido'].std():.4f}")

# By category
print("\n📈 BY NOISE CATEGORY:")
grouped = df.groupby('categoria_ruido')['ruido'].agg(['count', 'mean', 'std', 'min', 'max'])
print(grouped)

print("\n📋 DETAILED BREAKDOWN:")
high_noise = df[df['categoria_ruido'] == 'alto']
low_noise = df[df['categoria_ruido'] == 'bajo']

print(f"\n  HIGH NOISE (NSR > {df['ruido'].median():.2f}):")
print(f"    Count: {len(high_noise)} ({len(high_noise)/len(df)*100:.1f}%)")
print(f"    Mean NSR: {high_noise['ruido'].mean():.2f}")
print(f"    Std Dev: {high_noise['ruido'].std():.2f}")

print(f"\n  LOW NOISE (NSR ≤ {df['ruido'].median():.2f}):")
print(f"    Count: {len(low_noise)} ({len(low_noise)/len(df)*100:.1f}%)")
print(f"    Mean NSR: {low_noise['ruido'].mean():.2f}")
print(f"    Std Dev: {low_noise['ruido'].std():.2f}")

ratio = high_noise['ruido'].mean() / low_noise['ruido'].mean()
print(f"\n  Ratio (High/Low): {ratio:.2f}x")

# Distribution by noise and change categories
print("\n📊 DISTRIBUTION BY NOISE AND CHANGE CATEGORIES:")
distribution = df.groupby(['categoria_ruido', 'categoria_cambio']).size().unstack(fill_value=0)
print(distribution)
print(f"\nTotal combinations: {distribution.sum().sum()}")

print("\n" + "=" * 80)
print("✓ Statistics calculated successfully")
print("=" * 80)
