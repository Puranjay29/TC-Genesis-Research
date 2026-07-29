#!/usr/bin/env python3
import os
import sys

log_file = "run_xai_attribution.py"

if not os.path.exists(log_file):
    print(f"❌ Error: {log_file} not found!")
    sys.exit(1)

with open(log_file, 'r') as f:
    code = f.read()

# Swap out the invalid color string for a native standard hex representation
fixed_code = code.replace("color='emerald'", "color='#2ecc71'")

with open(log_file, 'w') as f:
    f.write(fixed_code)

print("✅ run_xai_attribution.py successfully updated to use standard green color hex.")
print("Modifying script to print out metrics directly...")

# Let's bypass the full evaluation data matrix load to verify text tracking limits instantly
# By executing a safe dummy run of step 4 manually to extract the numeric matrix array
