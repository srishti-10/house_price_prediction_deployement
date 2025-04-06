# House Price Prediction Project

This project implements a machine learning model for predicting house prices. It includes two implementations:
1. A modular implementation in the `house_prices` directory
2. A packaged implementation in the `house_prices_package` directory

## Project Structure

```
dsp_srishti_binwani_pw2/
├── data/                      # Data directory (not included in repo)
├── models/                    # Saved model artifacts
├── notebooks/                 # Jupyter notebooks
│   └── model-industrialization-final.ipynb
├── house_prices/             # Modular implementation
│   ├── __init__.py
│   ├── preprocess.py         # Data preprocessing functionality
│   ├── train.py             # Model training functionality
│   └── inference.py         # Model inference functionality
└── house_prices_package/     # Package implementation
    ├── setup.py             # Package setup configuration
    ├── requirements.txt     # Package dependencies
    ├── README.md           # Package documentation
    └── house_prices/       # Package source code
        ├── __init__.py
        ├── preprocess.py
        ├── train.py
        └── inference.py
```

## Implementation 1: Modular Code

The modular implementation in `house_prices/` provides direct access to the model functionality. It includes:

- Data preprocessing (feature selection, scaling, encoding)
- Model training with RandomForestRegressor
- Model inference capabilities
- Type hints and comprehensive docstrings
- PEP 8 compliant code

### Usage (Modular)

```python
from house_prices import train, inference, preprocess

# Train model
model_metrics = train.build_model(training_data)

# Make predictions
predictions = inference.make_predictions(new_data)
```

## Implementation 2: Python Package

The package implementation in `house_prices_package/` provides the same functionality in a properly packaged format, making it easy to install and distribute.

### Installation

To install the package in development mode:

```bash
cd house_prices_package
pip install -e .
```

### Usage (Package)

```python
from house_prices import train, inference

# Train model
model_metrics = train.build_model(training_data)

# Make predictions
predictions = inference.make_predictions(new_data)
```

## Features

Both implementations include:
- Feature selection for both numeric and categorical variables
- Missing value imputation
- Feature scaling
- One-hot encoding for categorical variables
- Model training with cross-validation
- Model persistence
- Prediction capabilities

## Dependencies

- Python >= 3.6
- pandas
- numpy
- scikit-learn
- joblib

## Code Quality

The code follows best practices including:
- Type hints for better code readability and IDE support
- Comprehensive docstrings following standard Python conventions
- PEP 8 compliance (verified with flake8)
- Modular design with clear separation of concerns
- Error handling and input validation

## Author

Srishti Binwani
