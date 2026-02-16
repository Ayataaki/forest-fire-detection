# Forest Fire Detection 

This project explores satellite imagery to detect and analyze areas affected by wildfires in Canada.  
It includes scripts for dataset ingestion from Kaggle, preprocessing, and exploratory notebooks for visualization and analysis.

## Project Structure
- `data/raw/wildfire/` : Raw dataset downloaded from Kaggle
- `scripts/` : Utility scripts (e.g., dataset download)
- `notebooks/` : Jupyter notebooks for exploration and modeling
- `requirements.txt` : Python dependencies

## Installation
Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```


---

### 4. Dataset Ingestion
```markdown
## Dataset Ingestion
To download the wildfire dataset:
```

```bash
python scripts/download_wildfire.py
```


---

### 5. Data Exploration

## Data Exploration
Start Jupyter Notebook and open:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```


---

### 6. Usage Example
## Usage Example
- Run `scripts/download_wildfire.py` to fetch data.
- Explore images in `notebooks/01_data_exploration.ipynb`.
- Extend notebooks for preprocessing, feature extraction, and model training.


## License
MIT License

## Credits
Dataset: [Wildfire Prediction Dataset (Satellite Images)](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset)
