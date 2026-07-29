#!/usr/bin/env python3
import os
import sys
import pandas as pd

# The original locked path and our clean rescue path
old_log = "data/extracted_features_entire_world/test_predictions_log.csv"
rescue_log = "data/extracted_features_entire_world/test_predictions_log_v2.csv"

print("--- RECOVERING LOGS TO ACCESSIBLE RECOVERY PATH ---")
try:
    # Overwrite the plot coordinates script to point to the new clean log file
    with open('plot_coordinates_deviation.py', 'r') as f:
        plot_code = f.read()
    
    # Switch the reference over to v2 dynamically
    plot_code = plot_code.replace(old_log, rescue_log)
    
    with open('plot_coordinates_deviation.py', 'w') as f:
        f.write(plot_code)
        
    print("✅ plot_coordinates_deviation.py successfully updated to point to v2 log asset.")
except Exception as e:
    print(f"⚠️ Tracking update warning: {e}")

# Let's generate a wrapper script that runs the logging code using the clean v2 file output path
with open('log_test_predictions.py', 'r') as f:
    log_code = f.read()

# Swap output path string out to prevent Windows sharing lock crashes
log_code = log_code.replace(f'output_log_path = "{old_log}"', f'output_log_path = "{rescue_log}"')

with open('log_test_predictions_v2.py', 'w') as f:
    f.write(log_code)

print("\n🚀 Ready! Close Excel if you want, but you don't even have to.")
print("Run the command below to write the file cleanly to v2 and view coordinates instantly:\n")
print("python3 log_test_predictions_v2.py && python3 plot_coordinates_deviation.py\n")
