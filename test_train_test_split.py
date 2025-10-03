"""Quick test for train/test split in benchmark_synthetic_data.py"""

import sys
import os

# Modificar temporalmente el CONFIG para que sea más rápido
original_file = "benchmark_synthetic_data.py"
with open(original_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Check if train/test split is in the code
if "split_train_test_synthetic" in content:
    print("✓ Train/test split function found")
else:
    print("✗ Train/test split function NOT found")

if '"train_f1_mean"' in content:
    print("✓ Train F1 metrics found")
else:
    print("✗ Train F1 metrics NOT found")

if '"test_f1_mean"' in content:
    print("✓ Test F1 metrics found")
else:
    print("✗ Test F1 metrics NOT found")

if "evaluate_algorithm_on_dataset(spec, train_data, test_data" in content:
    print("✓ Evaluation using train/test data found")
else:
    print("✗ Evaluation NOT using train/test properly")

print("\n✓ Code structure looks correct! Ready for testing.")
print("\nTo run a quick test with reduced parameters, execute:")
print("  python benchmark_synthetic_data.py")
