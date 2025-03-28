import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error
import joblib
import os
from models.preprocessing import HousePricePreprocessor

def train_and_save_model():
    # Load data
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    # Initialize preprocessor
    preprocessor = HousePricePreprocessor()

    # Prepare features and target
    X = train_df.drop(['SalePrice', 'Id'], axis=1)
    y = train_df['SalePrice']

    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit and transform training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    # Train the model
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train_processed, y_train)

    # Save the preprocessor and model
    preprocessor.save('models/preprocessor.joblib')
    joblib.dump(model, 'models/model.joblib')
    print("Model saved to models/model.joblib")

    # Calculate validation score
    val_predictions = model.predict(X_val_processed)
    val_score = np.sqrt(mean_squared_log_error(y_val, val_predictions))
    print(f"Validation RMSLE: {val_score:.4f}")

if __name__ == "__main__":
    train_and_save_model()
