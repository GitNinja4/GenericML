# genericML

## Project Overview
genericML is a modular Streamlit application for interactive end-to-end machine learning workflows.

## Features
- CSV dataset upload and validation
- Dataset inspection and exploratory data analysis
- User-controlled preprocessing
- Registry-driven model selection, per-model configuration, and training
- Train/test splitting before fitting preprocessing to avoid leakage
- Classification and regression workflows through Phase 5
- Tuning, evaluation, MLflow, and prediction pages are placeholders; their backends are not implemented

## Architecture
```
User
  │
  ▼
Streamlit UI
  │
  ├── Dataset management
  ├── EDA and feature inspection
  ├── Preprocessing configuration
  ├── Model selection and training
  ├── Tuning and evaluation
  ├── MLflow tracking
  └── Prediction pipeline
  │
  ▼
Python backend modules under src/
  │
  ├── data
  ├── preprocessing
  ├── models
  ├── training
  ├── tuning
  ├── evaluation
  ├── visualization
  ├── mlflow
  ├── prediction
  └── utils
```

## Tech Stack
- Python
- Streamlit
- Pandas
- NumPy
- scikit-learn
- Plotly
- Joblib
- pytest

Optuna, MLflow, and imbalanced-learn are reserved dependencies for future phases; the current app does not import or enable those integrations.

## Folder Structure
```
genericML/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── config/
├── src/
├── ui/
├── artifacts/plots/
├── data/
├── tests/
└── .venv/
```

## Installation
1. Create a virtual environment.
2. Activate it.
3. Install dependencies.

## Virtual Environment Setup
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Application
```bash
streamlit run app.py
```

## How to Use the Application
1. Upload a CSV dataset and select a target/problem type.
2. Review exploratory analysis and dataset diagnostics.
3. Configure and apply preprocessing.
4. Select compatible models and configure their parameters.
5. Configure the train/test split and random state.
6. Train selected models and inspect per-model training results.

## Model Library
The model library is generated from `src/models/registry.py`. Available models and family filters follow the selected classification or regression problem type; the UI does not maintain a separate model list.

## Model Selection and Training
- Available classification and regression models are discovered from `src/models/registry.py`.
- Search, family filtering, details, multi-selection, and individual parameter controls are provided by the Models page.
- User-controlled test size and random state
- Training reuses the Phase 4 preprocessing pipeline after the train/test split
- Complete preprocessing and model pipelines are retained in session state for the current session

## Application Navigation
- Session-driven sidebar navigation connects Dataset, EDA, Preprocessing, and Models.
- Dataset, EDA, Preprocessing, and Models expose guided workflow actions.
- Tuning, Evaluation, MLflow, and Prediction remain polished placeholders for future phases.

## Supported Preprocessing
- User-controlled missing-value strategies
- Leakage-safe IQR outlier handling
- One-hot and ordinal encoding with unseen-category handling
- Standard, MinMax, and Robust scaling options
- Optional SelectKBest feature selection
- Explicit feature exclusion and serializable preprocessing configuration

## Project Workflow
CSV Upload → Target Selection → EDA → Preprocessing → Model Selection → Training

Evaluation, MLflow, tuning, and prediction are future phases and remain disabled placeholders.

## Testing
```bash
pytest
```

## Troubleshooting
- Ensure a valid CSV is uploaded.
- Confirm the target column is selected.
- Verify the environment has all dependencies installed.

## Future Improvements
- Additional algorithms and feature engineering options
- Better model comparison dashboards
- Cloud deployment and CI/CD automation
```
