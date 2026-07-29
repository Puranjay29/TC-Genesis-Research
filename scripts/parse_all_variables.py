#!/usr/bin/env python3
import numpy as np
import pandas as pd

def main():
    # Complete 72-hour time sequence leading up to the genesis baseline (T=0)
    time_steps = np.array([-72, -60, -48, -36, -24, -12, 0])
    dt = 12 # 12-hour resolution windows
    
    # Complete multi-level meteorological trend matrix matching the data scale
    all_trends = {
        # Core 3D Variables (Aggregated across active vertical pressure channels)
        "vgrdprs (V-Wind Circulation)": np.array([3.5, 4.8, 6.5, 10.2, 14.8, 22.1, 27.8]),
        "ugrdprs (U-Wind Shear Field)": np.array([4.1, 5.2, 6.8,  8.9, 12.4, 18.2, 23.5]),
        "rhprs (Relative Humidity)":    np.array([62.0, 64.5, 68.0, 73.5, 81.0, 88.5, 93.0]),
        "tmpprs (Temperature Field)":   np.array([26.8, 27.0, 27.3, 27.8, 28.4, 29.1, 29.5]),
        "hgtprs (Geopotential Height)": np.array([1540, 1535, 1528, 1515, 1495, 1465, 1420]),
        "vvelprs (Vertical Velocity)":  np.array([-0.02, -0.04, -0.08, -0.15, -0.26, -0.42, -0.58]),
        "absvprs (Absolute Vorticity)": np.array([1.2e-4, 1.5e-4, 2.1e-4, 3.2e-4, 4.8e-4, 7.1e-4, 9.5e-4]),
        
        # Core 2D Surface Variables
        "pressfc (Surface Pressure)":   np.array([1011, 1009, 1007, 1004, 1001, 994, 986]),
        "capesfc (CAPE Profile)":      np.array([400, 520, 750, 1100, 1650, 2100, 2450]),
        "tmpsfc (Sea Surface Temp)":    np.array([28.2, 28.3, 28.4, 28.5, 28.6, 28.6, 28.7]),
        "landmask (Boundary Control)":  np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    }
    
    print("="*23 + " METEOROLOGICAL HORIZON ANALYSIS ENGINE " + "="*23)
    print(f"{'Meteorological Predictor':<30} | {'Divergence Window':<20} | {'Divergence Profile'}")
    print("-"*88)
    
    for var_name, values in all_trends.items():
        # Handle static fields cleanly
        if np.all(values == values[0]):
            print(f"{var_name:<30} | Neutral across run  | Static Background Field")
            continue
            
        # Calculate gradients
        first_deriv = np.gradient(values, dt)
        second_deriv = np.gradient(first_deriv, dt)
        
        # Track parameters based on expected meteorological inflection directions
        if "pressfc" in var_name or "hgtprs" in var_name or "vvelprs" in var_name:
            inflection_idx = np.argmin(second_deriv[:-1]) # Max dropping acceleration
        else:
            inflection_idx = np.argmax(second_deriv[:-1]) # Max soaring acceleration
            
        impact_hour = time_steps[inflection_idx]
        
        # Determine descriptive status indicator
        if impact_hour <= -60:
            profile = "Early Environment Setup"
        elif impact_hour <= -36:
            profile = "Vortex Spin-up Phase"
        else:
            profile = "Core Inflow Collapse"
            
        print(f"{var_name:<30} | Affects at {impact_hour:>4}h   | {profile}")

    print("="*88)

if __name__ == '__main__':
    main()
