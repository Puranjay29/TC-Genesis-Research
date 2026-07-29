#!/usr/bin/env python3
import os
import requests

# Constants & Configuration
API_TOKEN = "8ce29ab48a53fa8b70b99fc68095"
BASE_URL = "https://osdf-director.osg-htc.org/ncar/gdex/d083002/"
OUTPUT_DIR = "./data/ncep_fnl"

# Proxy handling configuration parsed explicitly for requests
PROXIES = {
    "http": "http://rrscnorth:NRSC%40User@192.168.0.9:8080",
    "https": "https://rrscnorth:NRSC%40User@192.168.0.9:8080"
}

# Adjust years/months based on the exact timeline you want to replicate
years = range(2016, 2023)  
months = [f"{m:02d}" for m in range(5, 12)]  # May to November (TC Season)
cycles = ["00", "06", "12", "18"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

headers = {
    "Authorization": f"Bearer {API_TOKEN}"
}

print("Starting download script...")

for year in years:
    for month in months:
        days_in_month = 30 if month in ["06", "09", "11"] else 31
        for day in range(1, days_in_month + 1):
            day_str = f"{day:02d}"
            for cycle in cycles:
                file_path = f"grib2/{year}/{year}.{month}/fnl_{year}{month}{day_str}_{cycle}_00.grib2"
                ofile_name = os.path.join(OUTPUT_DIR, f"fnl_{year}{month}{day_str}_{cycle}_00.grib2")
                
                if os.path.exists(ofile_name):
                    continue
                
                download_url = BASE_URL + file_path
                print(f"Downloading: {file_path}")
                
                try:
                    response = requests.get(download_url, headers=headers, proxies=PROXIES, timeout=30)
                    if response.status_code == 200:
                        with open(ofile_name, "wb") as f:
                            f.write(response.content)
                    elif response.status_code == 404:
                        continue
                    else:
                        print(f"Failed to fetch {file_path}: HTTP {response.status_code}")
                except Exception as e:
                    print(f"Error transferring {file_path}: {e}")