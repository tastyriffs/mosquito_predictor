# DC Mosquito Predictor

A machine learning model that predicts high mosquito abundance at trap sites across Washington, DC using weather, topography, and seasonal features.

## Overview

This project uses XGBoost to classify mosquito trap collections as **High abundance** (≥ 23 females collected) or **Low abundance**, based on data from DC mosquito trap sites merged with NOAA weather records and LiDAR-derived terrain data.

**High abundance** is defined using the IQR 1.5× outlier threshold on female mosquito counts.

## Data Sources

| File | Description |
|---|---|
| `Mosquito_Trap_Sites.csv` | DC mosquito trap collection records |
| `Mosquito_Weather_NOAA.csv` | Daily weather data from NOAA |
| `DTM.tif` | LiDAR Digital Terrain Model for DC |

## Features

The model uses 27 features across four categories:

- **Spatial** — trap latitude, longitude, and Topographic Wetness Index (TWI)
- **Weather** — daily avg/max/min temp, precipitation, 7/14/21-day lagged values, and rolling averages
- **Engineered** — cumulative degree days, wet day sequences, days since rain, temp × precip interactions, and temperature range
- **Seasonal** — day of year, week, month, and collection time of day

Precipitation and TWI features are log-transformed (log1p) to reduce skew.

## Model

**XGBoost Classifier** trained with 5-fold Group Cross-Validation (grouped by trap site to prevent data leakage).

Key hyperparameters:
```
scale_pos_weight = 12   # handles class imbalance
max_depth        = 3
learning_rate    = 0.05
n_estimators     = 127  # avg best iteration from CV
prob_threshold   = 0.30 # classification cutoff, tuned for recall
```

**Performance** (5-fold CV, IQR 1.5× threshold):
- ROC-AUC: ~0.72
- Macro F1: above stratified random baseline (0.481)
- Optimized for recall to minimize missed high-abundance events

## Outputs

The trained model is saved as:
- `mosquito_model.pkl` — XGBoost classifier
- `feature_cols.pkl` — ordered feature list for inference

## Usage

```python
from predict import predict_abundance

prob = predict_abundance(
    latitude=38.920,
    longitude=-77.030,
    collect_time_of_day='Morning',
    tavg=28.0, tmax=33.0, tmin=23.0,
    prcp=15.0,
    # ... (see notebook for full parameter list)
)

print(f"Probability of High abundance: {prob}")
# e.g. 0.82 → 82% chance of ≥ 23 females collected
```

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
optuna
folium
rasterio
pysheds
pyproj
geopandas
shapely
joblib
```

Install with:
```bash
pip install -r requirements.txt
```

## Notebook Structure

1. **Load & Clean** — trap site data and NOAA weather data
2. **LiDAR TWI** — compute Topographic Wetness Index from DTM raster
3. **EDA** — trap site maps, dispersion plots, feature distributions
4. **Feature Engineering** — labeling, log transforms, new interaction features
5. **Modeling** — XGBoost baseline, threshold exploration, feature importance
6. **Hyperparameter Tuning** — Optuna-based search with F1 floor constraint
7. **Final Model** — training, saving, and prediction function
