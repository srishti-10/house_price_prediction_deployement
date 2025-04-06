# House Price Prediction Model

A machine learning package for predicting house prices using gradient boosting regression. The model is built with scikit-learn and follows best practices for code organization, type hinting, and documentation.

## Project Structure

```
project/
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   ├── house_prices/
│   │   ├── __init__.py
│   │   ├── preprocess.py
│   │   ├── train.py
│   │   └── inference.py
│   └── model-industrialization-final.ipynb
└── models/
    ├── model.joblib
    └── preprocessor.joblib
```

## Features

- Data preprocessing pipeline with:
  - Missing value imputation
  - Feature scaling
  - Categorical encoding
- Gradient Boosting Regression model
- Type hints and comprehensive docstrings
- Modular code organization
- PEP8 compliant

## Requirements

- Python 3.9+
- pandas
- numpy
- scikit-learn
- joblib

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Model Performance

The model is evaluated using Root Mean Squared Logarithmic Error (RMSLE). Current performance metrics:
- RMSLE: ~0.159 on validation set

## Code Quality

- Type hints for better code understanding and IDE support
- Comprehensive docstrings following standard format
- PEP8 compliant code style
- Modular design with separate preprocessing, training, and inference components
