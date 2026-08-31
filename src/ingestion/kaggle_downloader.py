"""
Ingestion des images wildfire depuis Kaggle
Dataset: elmadafri/the-wildfire-dataset
"""

import os
import zipfile
import subprocess

DATASET = "abdelghaniaaba/wildfire-prediction-dataset"
RAW_DIR = "../data/raw/wildfire"

def download_dataset():
    os.makedirs(RAW_DIR, exist_ok=True)

    print("Téléchargement du dataset Kaggle...")
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", DATASET,
        "-p", RAW_DIR,
        "--unzip"
    ], check=True)

    print("Téléchargement terminé.")

if __name__ == "__main__":
    download_dataset()
