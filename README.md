

# F1 Podium Prediction Pipeline

A reproducible end-to-end machine learning workflow that ingests Formula 1 telemetry and qualifying data via the FastF1 API, trains an XGBoost model to predict race podium finishers, and exposes real-time predictions through a Streamlit dashboard.

---

## Table of Contents

1. [Features](#features)  
2. [Getting Started](#getting-started)  
   - [Prerequisites](#prerequisites)  
   - [Installation](#installation)  
3. [Data Acquisition & Preprocessing](#data-acquisition--preprocessing)  
4. [Feature Engineering](#feature-engineering)  
5. [Model Training & Evaluation](#model-training--evaluation)  
6. [Dashboard Deployment](#dashboard-deployment)  
7. [Use Cases](#use-cases)  
8. [Future Enhancements](#future-enhancements)  
9. [License](#license)  

---

## Features

- **FastF1 integration**  
  Caches and pulls session data (practice, qualifying, race) for all seasons from 2010 to present, with automatic local caching to speed up repeated queries.

- **Automated feature extractor**  
  Computes per-driver and per-team statistics (lap times, sector splits, qualifying ranks) and merges with championship standings.

- **High-performance model**  
  XGBoost classifier achieving 72 % podium finish accuracy and inference latency under 100 ms, validated against a 57 % logistic regression baseline.

- **Interactive dashboard**  
  Streamlit app that lets you select season, circuit, and session, then visualizes top-3 podium predictions with SHAP feature importance plots for explainability.

---

## Getting Started

### Prerequisites

- Python 3.9 or higher  
- Git  
- Internet access for initial FastF1 data pulls  

### Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/f1-podium-predictor.git
   cd f1-podium-predictor

## Create and Activate a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows

## Install Dependencies

```bash
pip install -r requirements.txt

## Configure FastF1 Cache Directory

In your Python script or REPL:

```python
import fastf1
fastf1.Cache.enable_cache("path/to/cache")


## Data Acquisition & Preprocessing

### Download session data

Use FastF1 to fetch telemetry, lap, and timing data for each session.  
Data is automatically stored in the cache directory.

### Clean and merge

- Drop incomplete laps and sessions missing sector times  
- Merge lap records with driver standings and constructor mappings  
- Export consolidated data as `season_data.csv`

## Feature Engineering

- **Driver-level features**
  - Average lap time in final stint
  - Qualifying Q1, Q2, Q3 lap times and ranks
  - Sector 1, Sector 2, Sector 3 average splits

- **Team-level features**
  - Constructor points in current season
  - Two-race rolling average of team performance

- **Temporal features**
  - Session type (Practice, Qualifying, Race)
  - Track length and weather flags (dry vs wet)

All feature logic resides in `feature_extractor.py` with explicit column names.


## Model Training & Evaluation

### Train/test split

- Stratify by season and circuit to avoid data leakage.

### Model setup

- XGBoost classifier with early stopping on validation set.  
- Grid search for hyperparameters:  
  - `max_depth = 6`  
  - `eta = 0.1`  
  - `subsample = 0.8`  

### Performance metrics

- **Accuracy**: 72% podium prediction  
- **Inference latency**: < 100 ms per prediction  
- **Baseline**: 57% accuracy with logistic regression  

### Artifacts

- Save model to `models/model.pkl`  
- Save feature vectorizer to `models/features.pkl`  
- Export evaluation reports and confusion matrices to `reports/`  

python train_model.py --data season_data.csv --output-dir models/

## Dashboard Deployment

Launch the interactive Streamlit app:

```bash
streamlit run app.py


## Future Enhancements

- **Live telemetry streaming**: Integrate the official F1 live data feed for continuous, in-race prediction updates.
- **Weather data integration**: Automatically fetch track conditions and encode real-time weather metrics as features.
- **Ensemble models**: Combine XGBoost with LSTM or Temporal Convolutional Networks for sequential pattern learning.
- **Automated hyperparameter tuning**: Use Optuna to optimize accuracy and latency simultaneously.
- **Containerization**: Provide a Dockerfile for one-step setup and deployment across environments.

