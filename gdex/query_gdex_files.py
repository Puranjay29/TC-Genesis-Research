import urllib.request
import ssl
import json
import warnings
warnings.filterwarnings('ignore')

GDEX_TOKEN = "8ce29ab48a53fa8b70b99fc68095"
URL = "https://gdex.ucar.edu/api/v1/collections" # Query available data collections

headers = {
    "Authorization": f"Bearer {GDEX_TOKEN}",
    "User-Agent": "Mozilla/5.0"
}

try:
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(URL, headers=headers)
    with urllib.request.urlopen(req, context=ctx, timeout=15) as response:
        data = json.loads(response.read().decode('utf-8'))
        print("\n🔍 --- AVAILABLE NCAR DATASET COLLECTIONS REVEALED ---")
        print(json.dumps(data, indent=4))
        print("----------------------------------------------------\n")
except Exception as e:
    print(f"❌ Failed to parse registry index: {e}")
    print("💡 Let's try alternative endpoint mapping...")
    
    # Alternative user account endpoint check
    try:
        req = urllib.request.Request("https://gdex.ucar.edu/api/v1/user/datasets", headers=headers)
        with urllib.request.urlopen(req, context=ctx, timeout=15) as response:
            data = json.loads(response.read().decode('utf-8'))
            print(json.dumps(data, indent=4))
    except Exception as e2:
        print(f"❌ Secondary endpoint blocked: {e2}")
