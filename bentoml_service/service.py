import numpy as np
import pandas as pd
import bentoml
import os
import joblib

model_path = os.path.join("models", "house_price.pkl")

@bentoml.service(
    name="house_price_predictor",
    resources={"cpu": "1"}
)
class HousePricePredictor:
    def __init__(self):
        # Load the serialized model pipeline directly using joblib
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            self.model = None

    @bentoml.api
    def predict(
        self,
        area: int,
        bedrooms: int,
        bathrooms: int,
        stories: int,
        mainroad: str,
        guestroom: str,
        basement: str,
        hotwaterheating: str,
        airconditioning: str,
        parking: int,
        prefarea: str,
        furnishingstatus: str
    ) -> float:
        if self.model is None:
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
            else:
                raise Exception(f"Model file '{model_path}' not found. Please run train.py first.")
                
        # Format input feature records
        input_data = {
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "mainroad": mainroad,
            "guestroom": guestroom,
            "basement": basement,
            "hotwaterheating": hotwaterheating,
            "airconditioning": airconditioning,
            "parking": parking,
            "prefarea": prefarea,
            "furnishingstatus": furnishingstatus
        }
        input_df = pd.DataFrame([input_data])
        
        # Inferences
        predicted_arr = self.model.predict(input_df)
        
        # Return prediction float
        if hasattr(predicted_arr, 'iloc'):
            return float(predicted_arr.iloc[0])
        elif isinstance(predicted_arr, (list, pd.Series)) or hasattr(predicted_arr, '__len__'):
            return float(predicted_arr[0])
        else:
            return float(predicted_arr)
