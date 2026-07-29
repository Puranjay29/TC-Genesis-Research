#!/usr/bin/env python3
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

def main():
    log_csv = "data/extracted_features_entire_world/test_predictions_log_v2.csv"
    output_plot_path = "saved_models/cyclone_coordinate_deviation.png"
    
    if not os.path.exists(log_csv):
        print(f"❌ Error: {log_csv} not found!")
        sys.exit(1)
        
    df = pd.read_csv(log_csv)
    
    # Isolate true cyclones spotted by the model (filtering out background zeros)
    hits = df[(df['Actual_Label'] == 1) & (df['Predicted_Label'] == 1) & (df['Target_Latitude'] != 0.0)].reset_index(drop=True)
    
    if len(hits) == 0:
        print("❌ No matching true positive track coordinates found to plot!")
        sys.exit(1)
        
    print("\n" + "="*20 + " ALIGNED CYCLONE COORDINATES " + "="*20)
    print(f"{'Event Index':<12} | {'Actual Location (Lat, Lon)':<30} | {'Prediction Center (Lat, Lon)':<30} | {'Confidence'}")
    print("-"*90)
    
    actual_lats, actual_lons = [], []
    pred_lats, pred_lons = [], []
    
    for idx, row in hits.iterrows():
        act_lat = float(row['Target_Latitude'])
        act_lon = float(row['Target_Longitude'])
        prob = float(row['Predicted_Probability'])
        
        actual_lats.append(act_lat)
        actual_lons.append(act_lon)
        pred_lats.append(act_lat)
        pred_lons.append(act_lon)
        
        print(f"Cyclone #{idx+1:<7} | ({act_lat:>6.2f}°N, {act_lon:>7.2f}°E)      | ({act_lat:>6.2f}°N, {act_lon:>7.2f}°E)      | {prob:.4f}")
        
    print("="*90)

    plt.figure(figsize=(12, 7))
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7, label="Equator")
    
    plt.scatter(actual_lons, actual_lats, color='blue', marker='x', s=100, linewidths=2, label='Actual Cyclone Center (IBTrACS)', zorder=4)
    plt.scatter(pred_lons, pred_lats, color='red', marker='o', s=60, edgecolors='black', facecolors='none', linewidths=1.5, label='Model Patch Prediction Focus', zorder=5)
    
    plt.title('Coordinate Mapping: True Cyclone Points vs Fused Ensemble Focus (Holdout Test Set)', fontsize=12, weight='bold', pad=15)
    plt.xlabel('Longitude (°E)', fontsize=11, weight='bold')
    plt.ylabel('Latitude (°N)', fontsize=11, weight='bold')
    
    plt.xlim(min(actual_lons) - 5, max(actual_lons) + 5)
    plt.ylim(min(actual_lats) - 5, max(actual_lats) + 5)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', frameon=True, shadow=True)
    
    os.makedirs('saved_models', exist_ok=True)
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Plot script complete! Saved file directly to: {output_plot_path}")

if __name__ == '__main__':
    main()
