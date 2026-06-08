import pandas as pd
import joblib
import os

def predict_single_record(features: dict):
    model_path = os.path.join("models", "house_price.pkl")
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found! Run train.py first.")
        return None
        
    # Load serialized pipeline using joblib
    model = joblib.load(model_path)
    if hasattr(model, "pipeline"):
        model = model.pipeline
    
    # Format input into a DataFrame
    input_df = pd.DataFrame([features])
    
    # Run prediction
    predicted_arr = model.predict(input_df)
    
    # Extract prediction value
    if hasattr(predicted_arr, 'iloc'):
        predicted_price = float(predicted_arr.iloc[0])
    elif isinstance(predicted_arr, (list, pd.Series)) or hasattr(predicted_arr, '__len__'):
        predicted_price = float(predicted_arr[0])
    else:
        predicted_price = float(predicted_arr)
        
    return predicted_price

if __name__ == "__main__":
    # Sample record for testing
    sample_features = {
        "area": 5000,
        "bedrooms": 3,
        "bathrooms": 2,
        "stories": 2,
        "mainroad": "yes",
        "guestroom": "no",
        "basement": "no",
        "hotwaterheating": "no",
        "airconditioning": "yes",
        "parking": 1,
        "prefarea": "no",
        "furnishingstatus": "semi-furnished"
    }
    
    print("Running sample inference check...")
    price = predict_single_record(sample_features)
    if price is not None:
        print(f"Predicted House Value: INR {price:,.2f}")
