import os
import urllib.request
import ssl
import json
import warnings

warnings.filterwarnings("ignore")

GDEX_TOKEN = os.getenv("GDEX_TOKEN")

if not GDEX_TOKEN:
    raise RuntimeError(
        "GDEX_TOKEN environment variable is not set. "
        "Set it before running this script."
    )

URL = "https://gdex.ucar.edu/api/v1/collections"

headers = {
    "Authorization": f"Bearer {GDEX_TOKEN}",
    "User-Agent": "Mozilla/5.0"
}

ctx = ssl.create_default_context()

try:
    req = urllib.request.Request(URL, headers=headers)

    with urllib.request.urlopen(req, context=ctx, timeout=15) as response:
        data = json.loads(response.read().decode("utf-8"))

    print("\n🔍 --- AVAILABLE NCAR DATASET COLLECTIONS REVEALED ---")
    print(json.dumps(data, indent=4))
    print("----------------------------------------------------\n")

except Exception as e:
    print(f"❌ Failed to query registry: {e}")
    print("💡 Trying alternative endpoint...")

    try:
        fallback_url = "https://gdex.ucar.edu/api/v1/user/datasets"
        req = urllib.request.Request(fallback_url, headers=headers)

        with urllib.request.urlopen(req, context=ctx, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))

        print("\n🔍 --- USER DATASETS ---")
        print(json.dumps(data, indent=4))
        print("-----------------------\n")

    except Exception as e2:
        print(f"❌ Secondary endpoint failed: {e2}")