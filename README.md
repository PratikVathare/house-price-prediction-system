# House Price Prediction System

This is the simplest, production-ready form of the House Price Prediction System. It includes regression modeling (PyCaret AutoML), tracking (MLflow), serving (BentoML), REST API (FastAPI), human interface (Flask), database logging, and CI/CD config (Jenkins).

---

## Folder Structure

```
house-price-prediction/
├── data/
│   └── Housing.csv              # Kaggle raw dataset
├── models/
│   └── house_price.pkl          # Serialized PyCaret model (created after training)
├── src/
│   ├── automl.py                # PyCaret model search comparison script
│   ├── mlflow_tracking.py       # MLflow logging test script
│   ├── train.py                 # AutoML model training & serialization script
│   └── predict.py               # Local inference test script
├── flask_app/
│   ├── app.py                   # Flask Web Application UI
│   └── templates/
│       └── index.html           # Simple responsive input form page
├── fastapi_app/
│   └── main.py                  # FastAPI REST API + SQLAlchemy logs auditor
├── database/
│   └── schema.sql               # MySQL Workbench database creation query script
├── bentoml_service/
│   └── service.py               # BentoML serving code
├── jenkins/
│   └── Jenkinsfile              # CI/CD pipeline script
└── requirements.txt             # Python packages listing
```

---

## How to Run the System (Step-by-Step)

### 1. Setup Virtual Environment
Run standard Python virtualenv activation commands:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# Install requirements
pip install -r requirements.txt --pre --no-cache-dir
```

### 2. Create the Database (Optional)
If using **MySQL Workbench**, copy and execute the query script in [database/schema.sql](file:///d:/BTech/sixth%20sem/AWDL/External/House%20Price%20Prediction%20System/database/schema.sql) to provision the database. 
*Note: If no MySQL connection is detected, the services will automatically and silently fall back to a local SQLite database (`predictions.db`) to enable immediate execution.*

### 3. Run AutoML Training & Track (MLflow)
Train the model, run AutoML, log experiment metrics to MLflow, and serialize the pipeline to the models folder:
```bash
python src/train.py
```
To run the MLflow Experiment Tracker Dashboard:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
*Navigate to `http://localhost:5000` to view experiments.*

### 4. Start the REST API Service (FastAPI)
Spin up the FastAPI predictor and log auditor service:
```bash
uvicorn fastapi_app.main:app --port 8000 --reload
```
*Access the auto-generated Swagger UI testing page at `http://localhost:8000/docs`.*

### 5. Start the Web Application (Flask)
Launch the Flask web user interface:
```bash
python flask_app/app.py
```
*Open `http://localhost:5000` in your browser to test prediction forms.*

### 6. BentoML Serve
Verify BentoML serving configurations:
```bash
bentoml serve bentoml_service/service:HousePricePredictor --port 3000 --reload
```
