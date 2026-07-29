#!/usr/bin/env python3
"""
Geographical Visualization Script for TC Genesis Model Performance.
Plots Tropical Cyclogenesis Genesis Events and Model Classification Performance 
across Global Ocean Basins (Atlantic, East Pacific, West Pacific, Indian Ocean).
"""

import os
import sys
import ssl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Disable SSL certificate verification for Cartopy background map downloads through corporate proxy
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass


def main():
    log_file = "data/extracted_features_entire_world/test_predictions_log_24h.csv"
    if not os.path.exists(log_file):
        log_file = "test_predictions_log_24h.csv"
        
    df = pd.read_csv(log_file)
    print(f"Loaded prediction log: {log_file} ({len(df)} records)")
    
    # Filter valid coordinates
    df_valid = df[(df['Target_Latitude'] != 'N/A') & (df['Target_Longitude'] != 'N/A')].copy()
    df_valid['Target_Latitude'] = pd.to_numeric(df_valid['Target_Latitude'], errors='coerce')
    df_valid['Target_Longitude'] = pd.to_numeric(df_valid['Target_Longitude'], errors='coerce')
    df_valid = df_valid.dropna(subset=['Target_Latitude', 'Target_Longitude'])
    
    print(f"Valid coordinate records: {len(df_valid)}")
    
    # Convert longitude range if needed (-180 to 180)
    df_valid['Lon_Plot'] = df_valid['Target_Longitude'].apply(lambda x: x - 360 if x > 180 else x)
    
    # Categorize Performance Outcome
    def get_outcome(row):
        actual = row['Actual_Label']
        pred = row['Pred_Ensemble_Class']
        if actual == 1 and pred == 1:
            return 'True Positive (Genesis Caught)'
        elif actual == 1 and pred == 0:
            return 'False Negative (Missed Genesis)'
        elif actual == 0 and pred == 1:
            return 'False Positive (False Alarm)'
        else:
            return 'True Negative (Correct Ocean Background)'
            
    df_valid['Outcome'] = df_valid.apply(get_outcome, axis=1)
    
    # Setup Figure with Ocean Basin Map Background
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        plt.close(fig)
        
        fig = plt.figure(figsize=(18, 10), dpi=300)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
        ax.add_feature(cfeature.LAND, facecolor='#e0e0e0', edgecolor='gray')
        ax.add_feature(cfeature.OCEAN, facecolor='#d4e6f1')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')
        ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.3)
        transform = ccrs.PlateCarree()
    except ImportError:
        # Fallback if cartopy is not installed
        ax.set_facecolor('#d4e6f1')
        ax.set_xlim([-180, 180])
        ax.set_ylim([-60, 60])
        ax.set_xlabel('Longitude (°)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latitude (°)', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        transform = None

    # Define Category Colors and Markers
    style_map = {
        'True Positive (Genesis Caught)': {'color': '#1f77b4', 'marker': 'o', 'size': 60, 'label': 'True Positive'},
        'False Negative (Missed Genesis)': {'color': '#d62728', 'marker': 'X', 'size': 100, 'label': 'False Negative'},
        'False Positive (False Alarm)': {'color': '#ff7f0e', 'marker': '^', 'size': 90, 'label': 'False Positive'},
        'True Negative (Correct Ocean Background)': {'color': '#2ca02c', 'marker': 's', 'size': 30, 'label': 'True Negative'}
    }

    for category, style in style_map.items():
        sub = df_valid[df_valid['Outcome'] == category]
        if len(sub) > 0:
            if transform:
                ax.scatter(sub['Lon_Plot'], sub['Target_Latitude'],
                           color=style['color'], marker=style['marker'], s=style['size'],
                           edgecolor='black', linewidth=0.6, alpha=0.85,
                           transform=transform, label=f"{category} (n={len(sub)})")
            else:
                ax.scatter(sub['Lon_Plot'], sub['Target_Latitude'],
                           color=style['color'], marker=style['marker'], s=style['size'],
                           edgecolor='black', linewidth=0.6, alpha=0.85,
                           label=f"{category} (n={len(sub)})")

    plt.title("Gated Cross-Attention Ensemble Operational TC Genesis Performance Across Ocean Basins (24h Lead Time)",
              fontsize=14, fontweight='bold', pad=15)
              
    plt.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='black', fontsize=11)
    
    out_dir = "plots"
    os.makedirs(out_dir, exist_ok=True)
    out_img = os.path.join(out_dir, "spatial_model_performance_24h.png")
    plt.savefig(out_img, bbox_inches='tight')
    print(f"✅ Map plot successfully generated and saved to: {out_img}")


if __name__ == '__main__':
    main()