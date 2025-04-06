"""
House price prediction module with data preprocessing, model training, and inference functions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import joblib
import os

# Constants
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

TARGET_COLUMN = 'SalePrice'
MODEL_DIR = 'models'


def create_preprocessor() -> ColumnTransformer:
    """
    Create a preprocessing pipeline for both numeric and categorical features.
    
    Returns:
        ColumnTransformer: The preprocessing pipeline
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'))
    ])

    return ColumnTransformer(transformers=[
        ('num', numeric_transformer, NUMERIC_FEATURES),
        ('cat', categorical_transformer, CATEGORICAL_FEATURES)
    ])


def prepare_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target from the input data.
    
    Args:
        data: Input DataFrame containing all features and target
        
    Returns:
        Tuple containing features DataFrame and target Series
    """
    features = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    target = data[TARGET_COLUMN] if TARGET_COLUMN in data.columns else None
    return features, target


def save_artifacts(preprocessor: Any, model: Any) -> None:
    """
    Save preprocessor and model artifacts to disk.
    
    Args:
        preprocessor: Fitted preprocessor object
        model: Trained model object
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, 'preprocessor.joblib'))
    joblib.dump(model, os.path.join(MODEL_DIR, 'model.joblib'))


def load_artifacts() -> Tuple[Any, Any]:
    """
    Load preprocessor and model artifacts from disk.
    
    Returns:
        Tuple containing preprocessor and model objects
    """
    preprocessor = joblib.load(os.path.join(MODEL_DIR, 'preprocessor.joblib'))
    model = joblib.load(os.path.join(MODEL_DIR, 'model.joblib'))
    return preprocessor, model


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate model performance metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        Dictionary containing metric names and values
    """
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
    return {'rmsle': rmsle}


def build_model(data: pd.DataFrame) -> Dict[str, float]:
    """
    Build and train the model, save artifacts, and return performance metrics.
    
    Args:
        data: Input DataFrame containing features and target
        
    Returns:
        Dictionary containing model performance metrics
    """
    # Prepare data
    features, target = prepare_data(data)
    X_train, X_val, y_train, y_val = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    
    # Create and fit preprocessor
    preprocessor = create_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    
    # Train model
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train_processed, y_train)
    
    # Save artifacts
    save_artifacts(preprocessor, model)
    
    # Calculate and return metrics
    val_predictions = model.predict(X_val_processed)
    return calculate_metrics(y_val, val_predictions)


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    """
    Make predictions on new data using saved model and preprocessor.
    
    Args:
        input_data: DataFrame containing features for prediction
        
    Returns:
        Array of predictions
    """
    # Load artifacts
    preprocessor, model = load_artifacts()
    
    # Prepare and transform features
    features, _ = prepare_data(input_data)
    X_processed = preprocessor.transform(features)
    
    # Make predictions
    return model.predict(X_processed)
