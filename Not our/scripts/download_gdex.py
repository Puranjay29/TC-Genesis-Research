#!/usr/bin/env python3
import os
import requests
import urllib3
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress insecure proxy connection warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration & Core Parameters
API_TOKEN = "8ce29ab48a53fa8b70b99fc68095"
BASE_URL = "https://osdf-director.osg-htc.org/ncar/gdex/d083002/"
OUTPUT_DIR = "./data/ncep_fnl"

PROXIES = {
    "http": "http://rrscnorth:NRSC%40User@192.168.0.9:8080",
    "https": "http://rrscnorth:NRSC%40User@192.168.0.9:8080"
}

# The Paper's Exact Dataset Scope: 2008 through 2022
YEARS = range(2008, 2023)  
MONTHS = [f"{m:02d}" for m in range(5, 12)]  # May to November (Peak TC Season)
CYCLES = ["00", "06", "12", "18"]

# Optimization: Number of concurrent workers running through your 16-core CPU
MAX_WORKERS = 16  

os.makedirs(OUTPUT_DIR, exist_ok=True)

headers = {
    "Authorization": f"Bearer {API_TOKEN}"
}

def download_single_file(target_file):
    """Worker task to process a single GRIB2 file download session"""
    file_path = target_file["file_path"]
    ofile_name = target_file["ofile_name"]
    
    if os.path.exists(ofile_name):
        return f"Skipped (Exists): {ofile_name}"
        
    download_url = BASE_URL + file_path
    try:
        response = requests.get(download_url, headers=headers, proxies=PROXIES, verify=False, timeout=45)
        if response.status_code == 200:
            with open(ofile_name, "wb") as f:
                f.write(response.content)
            return f"Successfully Downloaded: {ofile_name}"
        elif response.status_code == 404:
            return f"Missing on Remote Server (404): {file_path}"
        else:
            return f"Failed HTTP {response.status_code}: {file_path}"
    except Exception as e:
        return f"Network Error on {file_path}: {e}"

def main():
    # Build complete list of all required targets across the paper's timeline
    download_queue = []
    for year in YEARS:
        for month in MONTHS:
            days_in_month = 30 if month in ["06", "09", "11"] else 31
            for day in range(1, days_in_month + 1):
                day_str = f"{day:02d}"
                for cycle in CYCLES:
                    file_path = f"grib2/{year}/{year}.{month}/fnl_{year}{month}{day_str}_{cycle}_00.grib2"
                    ofile_name = os.path.join(OUTPUT_DIR, f"fnl_{year}{month}{day_str}_{cycle}_00.grib2")
                    download_queue.append({"file_path": file_path, "ofile_name": ofile_name})

    print(f"Total target files mapped: {len(download_queue)}")
    print(f"Launching parallel execution engine using {MAX_WORKERS} concurrent proxy tunnels...")

    # Execute concurrent threaded distribution
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_download = {executor.submit(download_single_file, item): item for item in download_queue}
        
        for future in as_completed(future_to_download):
            result = future.result()
            # Suppress reporting skipped files to avoid printing thousands of lines
            if "Skipped" not in result:
                print(result)

if __name__ == "__main__":
    main()
