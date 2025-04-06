import pandas as pd
import joblib
from models.preprocessing import HousePricePreprocessor

def load_model_and_predict(data):
    """
    Load the saved model and preprocessor, then make predictions
    
    Args:
        data: pandas DataFrame with the same features as training data
    
    Returns:
        numpy array of predictions
    """
    # Load preprocessor and model
    preprocessor = HousePricePreprocessor.load('models/preprocessor.joblib')
    model = joblib.load('models/model.joblib')
    
    # Transform the data
    X_processed = preprocessor.transform(data)
    
    # Make predictions
    predictions = model.predict(X_processed)
    
    return predictions

if __name__ == "__main__":
    # Example usage
    test_df = pd.read_csv("data/test.csv")
    X_test = test_df.drop('Id', axis=1)
    
    predictions = load_model_and_predict(X_test)
    
    # Create submission
    submission = pd.DataFrame({
        'Id': test_df['Id'],
        'SalePrice': predictions
    })
    submission.to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")
