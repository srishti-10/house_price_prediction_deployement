"""Module for making predictions using the trained model."""

import pandas as pd
import numpy as np
import joblib
from .preprocess import preprocess_data, MODELS_DIR


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    """
    Make predictions on new data.

    Args:
        input_data: DataFrame containing features for prediction

    Returns:
        Array of predicted prices
    """
    # Load model
    model = joblib.load(f'{MODELS_DIR}/model.joblib')

    # Preprocess features
    X_processed = preprocess_data(input_data, is_training=False)

    # Make predictions
    predictions = model.predict(X_processed)

    return predictions
