"""Model training module for house price prediction."""

from typing import Dict
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

from .preprocess import preprocess_data, prepare_data, MODELS_DIR


def build_model(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict:
    """
    Build and train the model using the provided data.

    Args:
        data: DataFrame containing features and target
        test_size: Proportion of data to use for testing
        random_state: Random state for reproducibility

    Returns:
        Dictionary containing model performance metrics
    """
    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Split data into features and target
    features, target = prepare_data(data)

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )

    # Preprocess features
    X_train_processed = preprocess_data(X_train, is_training=True)
    X_val_processed = preprocess_data(X_val, is_training=False)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    model.fit(X_train_processed, y_train)

    # Save model
    joblib.dump(model, f'{MODELS_DIR}/model.joblib')

    # Calculate performance metrics
    train_score = model.score(X_train_processed, y_train)
    val_score = model.score(X_val_processed, y_val)

    return {
        'train_r2': train_score,
        'val_r2': val_score
    }
