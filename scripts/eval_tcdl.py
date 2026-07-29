import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

# ==========================================
# 1. Load IBTrACS Cyclone Tracks (1998-2026)
# ==========================================
def load_ibtracs_data(url="https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.ALL.list.v04r00.csv"):
    print("Loading IBTrACS global track data...")
    # Read headers and parse dates
    df = pd.read_csv(url, skiprows=[1], low_memory=False)
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
    df['SEASON'] = pd.to_numeric(df['SEASON'], errors='coerce')
    
    # Filter 1998 - 2026
    df = df[(df['SEASON'] >= 1998) & (df['SEASON'] <= 2026)].copy()
    df['USA_WIND'] = pd.to_numeric(df['USA_WIND'], errors='coerce')
    df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
    df['LON'] = pd.to_numeric(df['LON'], errors='coerce')
    
    return df

def extract_24h_prior_samples(df):
    """
    Finds cyclone genesis/events and shifts timestamps by -24 Hours (t - 24h).
    """
    samples = []
    grouped = df.groupby('SID')
    
    for sid, group in grouped:
        group = group.sort_values('ISO_TIME')
        if len(group) < 2:
            continue
            
        # First recorded time as Tropical Cyclone / Depression
        genesis_row = group.iloc[0]
        genesis_time = genesis_row['ISO_TIME']
        prior_time = genesis_time - pd.Timedelta(hours=24)
        
        samples.append({
            'sid': sid,
            'name': genesis_row['NAME'],
            'genesis_time': genesis_time,
            'prior_24h_time': prior_time,
            'lat': genesis_row['LAT'],
            'lon': genesis_row['LON'],
            'max_wind_kts': group['USA_WIND'].max()
        })
        
    return pd.DataFrame(samples)

# ==========================================
# 2. Model Inference Wrapper
# ==========================================
def run_tcdl_inference(model, samples_df, load_era5_feature_fn):
    """
    Parameters:
      model: Trained PyTorch/TF model for tc-dl
      samples_df: DataFrame with 24h prior timestamps and lat/lon
      load_era5_feature_fn: Function mapping (lat, lon, timestamp) -> ERA5 tensor
    """
    model.eval()
    results = []
    
    with torch.no_grad():
        for idx, row in samples_df.iterrows():
            # Extract 24h prior input tensor (e.g., shape [C, H, W] or feature vector)
            x_input = load_era5_feature_fn(row['lat'], row['lon'], row['prior_24h_time'])
            
            if x_input is None:
                continue
                
            input_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0) # Add batch dim
            
            # Forward pass
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).numpy()
            
            predicted_class = int(np.argmax(probs))
            confidence = float(probs[predicted_class])
            
            results.append({
                'SID': row['sid'],
                'Name': row['name'],
                'Genesis_Time': row['genesis_time'],
                'Prior_24h_Time': row['prior_24h_time'],
                'Predicted_Cyclone': bool(predicted_class == 1),
                'Confidence': confidence,
                'Prob_Cyclone': probs[1] if len(probs) > 1 else probs[0],
                'Actual_Max_Wind_kts': row['max_wind_kts']
            })
            
    return pd.DataFrame(results)

# ==========================================
# 3. Execution Setup
# ==========================================
if __name__ == "__main__":
    # 1. Fetch historical tracks
    tracks = load_ibtracs_data()
    eval_samples = extract_24h_prior_samples(tracks)
    print(f"Extracted {len(eval_samples)} cyclone genesis events (1998-2026) at t - 24h.")

    # Dummy tensor loader placeholder (replace with your local ERA5 xarray/netCDF loader)
    def mock_era5_loader(lat, lon, timestamp):
        # Return dummy array shaped to your tc-dl input specs, e.g., (channels, lat, lon)
        return np.random.randn(4, 64, 64)

    # 2. Load your tc-dl model weights
    # model = YourTCDLModel()
    # model.load_state_dict(torch.load("tcdl_weights.pth"))
    
    # Example execution structure:
    # results_df = run_tcdl_inference(model, eval_samples, mock_era5_loader)
    # results_df.to_csv("tc_dl_24h_predictions_1998_2026.csv", index=False)