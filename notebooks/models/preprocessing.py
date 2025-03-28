import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os

class HousePricePreprocessor:
    def __init__(self):
        # Initialize transformers
        self.numeric_features = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 
                               'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 
                               'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 
                               'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 
                               'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 
                               'ScreenPorch', 'PoolArea', 'MiscVal', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt']
        
        self.categorical_features = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
                                   'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 
                                   'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 
                                   'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond',
                                   'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                                   'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
                                   'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
                                   'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
                                   'SaleType', 'SaleCondition']

        # Create preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'))
        ])

        # Combine transformers
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])

    def fit_transform(self, X):
        """Fit the preprocessor and transform the data"""
        return self.preprocessor.fit_transform(X)

    def transform(self, X):
        """Transform new data using the fitted preprocessor"""
        return self.preprocessor.transform(X)

    def get_feature_names(self):
        """Get feature names after transformation"""
        numeric_features = self.numeric_features
        
        # Get categorical feature names after one-hot encoding
        cat_features = []
        onehot = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
        for i, feature in enumerate(self.categorical_features):
            feature_values = onehot.categories_[i][1:]  # Skip first category (dropped)
            cat_features.extend([f"{feature}_{val}" for val in feature_values])
        
        return numeric_features + cat_features

    def save(self, filepath='models/preprocessor.joblib'):
        """Save the preprocessor to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.preprocessor, filepath)
        print(f"Preprocessor saved to {filepath}")

    @classmethod
    def load(cls, filepath='models/preprocessor.joblib'):
        """Load a preprocessor from disk"""
        instance = cls()
        instance.preprocessor = joblib.load(filepath)
        print(f"Preprocessor loaded from {filepath}")
        return instance
