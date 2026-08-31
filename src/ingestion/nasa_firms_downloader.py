"""
NASA FIRMS Active Fire Data Downloader
Fetches real-time MODIS/VIIRS satellite fire detections
"""
import requests
import pandas as pd
import os

NASA_API_KEY = "8786a9c933621a88a4eec06dd722cdde"
BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

def download_firms(country="MAR", days=10, source="VIIRS_SNPP_NRT"):
    # Correct URL: BASE / KEY / SOURCE / COUNTRY / DAYS
    url = f"{BASE_URL}/{NASA_API_KEY}/{source}/{country}/{days}"
    print(f"Fetching: {url}")
    
    r = requests.get(url)
    
    if r.status_code != 200:
        print(f"Error {r.status_code}: {r.text}")
        return None

    os.makedirs("../data/raw/firms", exist_ok=True)
    path = "../data/raw/firms/firms_data.csv"
    
    with open(path, "w") as f:
        f.write(r.text)

    df = pd.read_csv(path)
    print(f"Downloaded {len(df)} fire detections")
    return df

if __name__ == "__main__":
    df = download_firms()
    if df is not None:
        print(df.head())
