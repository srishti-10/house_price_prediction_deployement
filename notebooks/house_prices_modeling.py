"""
House Price Prediction Model

This module contains functions for training and using a house price prediction model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error, mean_squared_error
import os
import joblib
from typing import Dict, Tuple, List

# Constants
MODELS_DIR = '../models'
NUMERIC_FEATURES = [
    'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 
    'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
    'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
    'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars',
    'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
    'ScreenPorch', 'PoolArea', 'MiscVal', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt'
]

CATEGORICAL_FEATURES = [
    'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
    'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
    'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
    'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond',
    'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
    'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
    'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
    'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
    'SaleType', 'SaleCondition'
]

def create_model_directory() -> None:
    """Create directory for storing model artifacts if it doesn't exist."""
    os.makedirs(MODELS_DIR, exist_ok=True)

def split_data(data: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and validation sets.
    
    Args:
        data: Input DataFrame with features and target
        target_column: Name of the target variable column
    
    Returns:
        Tuple of (X_train, X_val, y_train, y_val)
    """
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def preprocess_numeric_data(data: pd.DataFrame, is_training: bool = True) -> np.ndarray:
    """
    Preprocess numeric features with scaling.
    
    Args:
        data: DataFrame containing numeric features
        is_training: Whether this is for training or inference
    
    Returns:
        Scaled numeric features as numpy array
    """
    numeric_data = data[NUMERIC_FEATURES].copy()
    numeric_data = numeric_data.fillna(numeric_data.mean())
    
    if is_training:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        joblib.dump(scaler, f'{MODELS_DIR}/scaler.joblib')
    else:
        scaler = joblib.load(f'{MODELS_DIR}/scaler.joblib')
        scaled_data = scaler.transform(numeric_data)
    
    return scaled_data

def preprocess_categorical_data(data: pd.DataFrame, is_training: bool = True) -> np.ndarray:
    """
    Preprocess categorical features with one-hot encoding.
    
    Args:
        data: DataFrame containing categorical features
        is_training: Whether this is for training or inference
    
    Returns:
        Encoded categorical features as numpy array
    """
    categorical_data = data[CATEGORICAL_FEATURES].copy()
    categorical_data = categorical_data.fillna('Missing')
    
    if is_training:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_data = encoder.fit_transform(categorical_data)
        joblib.dump(encoder, f'{MODELS_DIR}/encoder.joblib')
    else:
        encoder = joblib.load(f'{MODELS_DIR}/encoder.joblib')
        encoded_data = encoder.transform(categorical_data)
    
    return encoded_data

def combine_features(numeric_data: np.ndarray, categorical_data: np.ndarray) -> pd.DataFrame:
    """Combine numeric and categorical features into a single DataFrame."""
    encoder = joblib.load(f'{MODELS_DIR}/encoder.joblib')
    feature_names = (NUMERIC_FEATURES + 
                    [f"{feat}_{val}" for feat, vals in zip(CATEGORICAL_FEATURES, encoder.categories_) 
                     for val in vals])
    return pd.DataFrame(np.hstack([numeric_data, categorical_data]), columns=feature_names)

def train_model(X: pd.DataFrame, y: pd.Series) -> LinearRegression:
    """Train a linear regression model and save it."""
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, f'{MODELS_DIR}/model.joblib')
    return model

def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics."""
    y_pred_positive = np.maximum(y_pred, 0)  # Ensure predictions are positive
    return {
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred_positive))),
        'rmsle': float(np.sqrt(mean_squared_log_error(y_true, y_pred_positive)))
    }

def build_model(data: pd.DataFrame) -> Dict[str, float]:
    """
    Build and evaluate the model.
    
    Args:
        data: Training data with target variable
    
    Returns:
        Dictionary containing model performance metrics
    """
    create_model_directory()
    
    # Split data
    X_train, X_val, y_train, y_val = split_data(data, target_column='SalePrice')
    
    # Preprocess features
    numeric_train = preprocess_numeric_data(X_train, is_training=True)
    categorical_train = preprocess_categorical_data(X_train, is_training=True)
    X_train_processed = combine_features(numeric_train, categorical_train)
    
    # Train model
    model = train_model(X_train_processed, y_train)
    
    # Process validation data and make predictions
    numeric_val = preprocess_numeric_data(X_val, is_training=False)
    categorical_val = preprocess_categorical_data(X_val, is_training=False)
    X_val_processed = combine_features(numeric_val, categorical_val)
    val_predictions = model.predict(X_val_processed)
    
    # Calculate and return metrics
    return calculate_metrics(y_val, val_predictions)

def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    """
    Make predictions on new data.
    
    Args:
        input_data: DataFrame containing features for prediction
    
    Returns:
        Array of predicted prices
    """
    # Preprocess features
    numeric_data = preprocess_numeric_data(input_data, is_training=False)
    categorical_data = preprocess_categorical_data(input_data, is_training=False)
    X_processed = combine_features(numeric_data, categorical_data)
    
    # Load model and make predictions
    model = joblib.load(f'{MODELS_DIR}/model.joblib')
    predictions = model.predict(X_processed)
    
    return np.maximum(predictions, 0)  # Ensure predictions are positive

if __name__ == '__main__':
    # Example usage
    train_data = pd.read_csv('../data/train.csv')
    test_data = pd.read_csv('../data/test.csv')
    
    # Build and evaluate model
    metrics = build_model(train_data)
    print("Model Performance:")
    print(f"RMSE: ${metrics['rmse']:.2f}")
    print(f"RMSLE: {metrics['rmsle']:.4f}")
    
    # Make predictions on test data
    test_predictions = make_predictions(test_data)
    
    # Save predictions
    submission = pd.DataFrame({
        'Id': test_data['Id'],
        'SalePrice': test_predictions
    })
    submission.to_csv('../data/submission.csv', index=False)
    print("\nPredictions saved to submission.csv")
