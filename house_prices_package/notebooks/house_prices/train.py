"""
House price model training module.

This module provides functions for training and evaluating the house price
prediction model. It handles model creation, training, and persistence.
"""

from typing import Dict, Any
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error
import joblib
from .preprocess import (
    prepare_data,
    create_preprocessor,
    save_preprocessor,
    MODEL_DIR,
    DataFrameType,
    ArrayType
)


def train_model(
    X_train: ArrayType,
    y_train: ArrayType,
    **kwargs: Any
) -> GradientBoostingRegressor:
    """
    Train a gradient boosting model on the processed data.

    Args:
        X_train: Processed feature matrix
        y_train: Target values
        **kwargs: Additional arguments for GradientBoostingRegressor

    Returns:
        Trained GradientBoostingRegressor model
    """
    default_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 4,
        'random_state': 42
    }
    params = {**default_params, **kwargs}
    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    return model


def save_model(
    model: GradientBoostingRegressor,
    filepath: str = None
) -> None:
    """
    Save trained model to disk.

    Args:
        model: Trained model to save
        filepath: Path to save model (default: models/model.joblib)
    """
    if filepath is None:
        filepath = os.path.join(MODEL_DIR, 'model.joblib')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)


def calculate_metrics(
    y_true: ArrayType,
    y_pred: ArrayType
) -> Dict[str, float]:
    """
    Calculate model performance metrics.

    Args:
        y_true: True target values
        y_pred: Predicted target values

    Returns:
        Dictionary containing metric names and values
    """
    rmsle = float(np.sqrt(mean_squared_log_error(y_true, y_pred)))
    return {'rmsle': rmsle}


def build_model(data: DataFrameType) -> Dict[str, float]:
    """
    Build and train the model, save artifacts, and return performance metrics.

    This function orchestrates the entire model building process:
    1. Prepares the data
    2. Creates and fits the preprocessor
    3. Trains the model
    4. Saves both preprocessor and model
    5. Evaluates model performance

    Args:
        data: Input DataFrame containing features and target

    Returns:
        Dictionary containing model performance metrics
    """
    features, target = prepare_data(data)
    X_train, X_val, y_train, y_val = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    preprocessor = create_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    model = train_model(X_train_processed, y_train)
    save_preprocessor(preprocessor)
    save_model(model)
    val_predictions = model.predict(X_val_processed)
    return calculate_metrics(y_val, val_predictions)
