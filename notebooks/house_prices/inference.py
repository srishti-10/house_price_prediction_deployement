"""
House price model inference module.

This module provides functions for making predictions using a trained
house price model. It handles loading the model and preprocessor.
"""

import os
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from .preprocess import (
    prepare_data,
    load_preprocessor,
    MODEL_DIR,
    DataFrameType,
    ArrayType
)


def load_model(filepath: str = None) -> GradientBoostingRegressor:
    """
    Load the trained model from disk.

    Args:
        filepath: Path to load the model from (default: models/model.joblib)

    Returns:
        Loaded GradientBoostingRegressor model
    """
    if filepath is None:
        filepath = os.path.join(MODEL_DIR, 'model.joblib')
    return joblib.load(filepath)


def make_predictions(input_data: DataFrameType) -> ArrayType:
    """
    Make predictions on new data using saved model and preprocessor.

    This function:
    1. Loads the saved preprocessor and model
    2. Prepares the input data
    3. Transforms the features using the preprocessor
    4. Makes predictions using the model

    Args:
        input_data: DataFrame containing features for prediction

    Returns:
        Array of predicted house prices
    """
    preprocessor = load_preprocessor()
    model = load_model()
    features, _ = prepare_data(input_data)
    X_processed = preprocessor.transform(features)
    return model.predict(X_processed)
