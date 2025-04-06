"""
House price data preprocessing module.

This module provides functions for preprocessing house price data, including:
- Feature selection
- Missing value imputation
- Scaling numerical features
- Encoding categorical features
"""

from typing import Tuple, List
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# Type aliases
PreprocessorType = ColumnTransformer
DataFrameType = pd.DataFrame
ArrayType = np.ndarray

# Constants
MODELS_DIR = '../models'

NUMERIC_FEATURES: List[str] = [
    'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
    'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
    'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
    'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars',
    'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
    'ScreenPorch', 'PoolArea', 'MiscVal', 'YearBuilt', 'YearRemodAdd',
    'GarageYrBlt'
]

CATEGORICAL_FEATURES: List[str] = [
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


def create_preprocessor() -> PreprocessorType:
    """
    Create a preprocessing pipeline for both numeric and categorical features.

    The pipeline includes:
    - For numeric features: median imputation and standard scaling
    - For categorical features: constant imputation and one-hot encoding

    Returns:
        PreprocessorType: A fitted sklearn ColumnTransformer object
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(
            drop='first',
            sparse=False,
            handle_unknown='ignore'
        ))
    ])

    return ColumnTransformer(transformers=[
        ('num', numeric_transformer, NUMERIC_FEATURES),
        ('cat', categorical_transformer, CATEGORICAL_FEATURES)
    ])


def prepare_data(data: DataFrameType) -> Tuple[DataFrameType, pd.Series]:
    """
    Prepare features and target from the input data.

    Args:
        data: Input DataFrame with features and target

    Returns:
        Tuple containing:
        - DataFrame with features
        - Series with target (None if target not in data)
    """
    features = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    target = data['SalePrice'] if 'SalePrice' in data.columns else None
    return features, target


def preprocess_data(
    data: DataFrameType,
    is_training: bool = True
) -> np.ndarray:
    """
    Preprocess data using the preprocessing pipeline.

    Args:
        data: DataFrame containing features
        is_training: Whether this is for training or inference

    Returns:
        Preprocessed features as numpy array
    """
    create_model_directory()
    features, _ = prepare_data(data)

    if is_training:
        preprocessor = create_preprocessor()
        processed_data = preprocessor.fit_transform(features)
        joblib.dump(preprocessor, f'{MODELS_DIR}/preprocessor.joblib')
    else:
        preprocessor = joblib.load(f'{MODELS_DIR}/preprocessor.joblib')
        processed_data = preprocessor.transform(features)

    return processed_data
