from flask import Flask, render_template, request, jsonify
import requests
import os
import joblib
import pandas as pd
import logging
from dotenv import load_dotenv

# Load environment configuration from .env
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

FASTAPI_URL = "http://localhost:8000"

@app.route("/")
def index():
    # Fetch history if FastAPI is available
    history = []
    try:
        r = requests.get(f"{FASTAPI_URL}/history", params={"limit": 10}, timeout=2)
        if r.status_code == 200:
            history = r.json().get("history", [])
    except Exception:
        logger.warning("Could not reach FastAPI to fetch history logs.")
        
    return render_template("index.html", history=history)

@app.route("/predict", methods=["POST"])
def predict():
    # Parse form inputs
    form_data = {
        "area": int(request.form.get("area", 0)),
        "bedrooms": int(request.form.get("bedrooms", 3)),
        "bathrooms": int(request.form.get("bathrooms", 1)),
        "stories": int(request.form.get("stories", 1)),
        "mainroad": request.form.get("mainroad", "no"),
        "guestroom": request.form.get("guestroom", "no"),
        "basement": request.form.get("basement", "no"),
        "hotwaterheating": request.form.get("hotwaterheating", "no"),
        "airconditioning": request.form.get("airconditioning", "no"),
        "parking": int(request.form.get("parking", 0)),
        "prefarea": request.form.get("prefarea", "no"),
        "furnishingstatus": request.form.get("furnishingstatus", "unfurnished")
    }
    
    # Try calling FastAPI prediction service
    try:
        response = requests.post(f"{FASTAPI_URL}/predict", json=form_data, timeout=3)
        if response.status_code == 200:
            result = response.json()
            return jsonify({
                "status": "success",
                "prediction": result.get("prediction"),
                "source": "FastAPI Service"
            })
    except Exception as e:
        logger.warning(f"FastAPI service unreachable ({e}). Falling back to local model inference...")
        
    # Local fallback model inference
    model_path = os.path.join("models", "house_price.pkl")
    if not os.path.exists(model_path):
        return jsonify({
            "status": "error",
            "message": "Model not trained yet! Run: python src/train.py"
        }), 503
        
    try:
        model = joblib.load(model_path)
        if hasattr(model, "pipeline"):
            model = model.pipeline
        input_df = pd.DataFrame([form_data])
        predicted_arr = model.predict(input_df)
        
        if hasattr(predicted_arr, 'iloc'):
            predicted_price = float(predicted_arr.iloc[0])
        elif isinstance(predicted_arr, (list, pd.Series)) or hasattr(predicted_arr, '__len__'):
            predicted_price = float(predicted_arr[0])
        else:
            predicted_price = float(predicted_arr)
            
        return jsonify({
            "status": "success",
            "prediction": predicted_price,
            "source": "Local Flask Model (Fallback)"
        })
    except Exception as local_err:
        return jsonify({
            "status": "error",
            "message": f"Local inference failed: {str(local_err)}"
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
