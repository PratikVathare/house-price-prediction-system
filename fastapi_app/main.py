from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import os
import joblib
import logging
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, text
from sqlalchemy.orm import declarative_base, sessionmaker

# Load environment configuration from .env
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="House Price Prediction API",
    description="FastAPI prediction engine with MySQL logging",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Database Configuration with Fallback
Base = declarative_base()

class HousePredictionRecord(Base):
    __tablename__ = "house_predictions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    area = Column(Integer)
    bedrooms = Column(Integer)
    bathrooms = Column(Integer)
    stories = Column(Integer)
    mainroad = Column(String(5))
    guestroom = Column(String(5))
    basement = Column(String(5))
    hotwaterheating = Column(String(5))
    airconditioning = Column(String(5))
    parking = Column(Integer)
    prefarea = Column(String(5))
    furnishingstatus = Column(String(20))
    predicted_price = Column(Float)
    source = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)

engine = None
SessionLocal = None

def init_db():
    global engine, SessionLocal
    # Try MySQL first (default credentials)
    db_user = os.getenv("DB_USER", "root")
    db_password = os.getenv("DB_PASSWORD", "password").strip("'\"")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "3306")
    db_name = os.getenv("DB_NAME", "housing_db")
    
    import urllib.parse
    safe_password = urllib.parse.quote_plus(db_password)
    
    try:
        temp_url = f"mysql+pymysql://{db_user}:{safe_password}@{db_host}:{db_port}"
        temp_engine = create_engine(temp_url, connect_args={"connect_timeout": 3})
        with temp_engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {db_name}"))
            conn.commit()
            
        mysql_url = f"mysql+pymysql://{db_user}:{safe_password}@{db_host}:{db_port}/{db_name}"
        engine = create_engine(mysql_url, connect_args={"connect_timeout": 3})
        with engine.connect() as conn:
            pass
        logger.info("Connected to MySQL successfully.")
    except Exception as e:
        logger.warning(f"MySQL connection failed: {e}. Falling back to SQLite.")
        engine = create_engine("sqlite:///predictions.db")
        
    SessionLocal = sessionmaker(bind=engine)
    Base.metadata.create_all(bind=engine)

init_db()

# 2. Input Schemas
class HouseFeatures(BaseModel):
    area: int = Field(..., json_schema_extra={"example": 5000})
    bedrooms: int = Field(..., json_schema_extra={"example": 3})
    bathrooms: int = Field(..., json_schema_extra={"example": 2})
    stories: int = Field(..., json_schema_extra={"example": 2})
    mainroad: str = Field(..., json_schema_extra={"example": "yes"})
    guestroom: str = Field(..., json_schema_extra={"example": "no"})
    basement: str = Field(..., json_schema_extra={"example": "no"})
    hotwaterheating: str = Field(..., json_schema_extra={"example": "no"})
    airconditioning: str = Field(..., json_schema_extra={"example": "yes"})
    parking: int = Field(..., json_schema_extra={"example": 1})
    prefarea: str = Field(..., json_schema_extra={"example": "no"})
    furnishingstatus: str = Field(..., json_schema_extra={"example": "semi-furnished"})

# 3. API Routes
@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    model_exists = os.path.exists(os.path.join("models", "house_price.pkl"))
    return {
        "status": "healthy",
        "model_available": model_exists,
        "database": engine.name if engine else "none"
    }

@app.post("/predict")
def predict(features: HouseFeatures):
    model_path = os.path.join("models", "house_price.pkl")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=503, detail="Model not trained yet! Run python src/train.py first.")
        
    try:
        model = joblib.load(model_path)
        if hasattr(model, "pipeline"):
            model = model.pipeline
        feature_dict = features.model_dump()
        input_df = pd.DataFrame([feature_dict])
        
        predicted_arr = model.predict(input_df)
        
        if hasattr(predicted_arr, 'iloc'):
            predicted_price = float(predicted_arr.iloc[0])
        elif isinstance(predicted_arr, (list, pd.Series)) or hasattr(predicted_arr, '__len__'):
            predicted_price = float(predicted_arr[0])
        else:
            predicted_price = float(predicted_arr)
            
        # Log to Database
        if SessionLocal:
            session = SessionLocal()
            try:
                record = HousePredictionRecord(
                    area=features.area,
                    bedrooms=features.bedrooms,
                    bathrooms=features.bathrooms,
                    stories=features.stories,
                    mainroad=features.mainroad,
                    guestroom=features.guestroom,
                    basement=features.basement,
                    hotwaterheating=features.hotwaterheating,
                    airconditioning=features.airconditioning,
                    parking=features.parking,
                    prefarea=features.prefarea,
                    furnishingstatus=features.furnishingstatus,
                    predicted_price=predicted_price,
                    source="FastAPI"
                )
                session.add(record)
                session.commit()
            except Exception as db_err:
                session.rollback()
                logger.error(f"Database logging failed: {db_err}")
            finally:
                session.close()
                
        return {
            "prediction": predicted_price,
            "currency": "INR",
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_history(limit: int = 50):
    if not SessionLocal:
        return {"history": []}
    session = SessionLocal()
    try:
        records = session.query(HousePredictionRecord).order_by(HousePredictionRecord.created_at.desc()).limit(limit).all()
        history = []
        for r in records:
            history.append({
                "id": r.id,
                "area": r.area,
                "bedrooms": r.bedrooms,
                "bathrooms": r.bathrooms,
                "stories": r.stories,
                "mainroad": r.mainroad,
                "guestroom": r.guestroom,
                "basement": r.basement,
                "hotwaterheating": r.hotwaterheating,
                "airconditioning": r.airconditioning,
                "parking": r.parking,
                "prefarea": r.prefarea,
                "furnishingstatus": r.furnishingstatus,
                "predicted_price": r.predicted_price,
                "source": r.source,
                "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else ""
            })
        return {"history": history}
    except Exception as e:
        logger.error(f"History fetch error: {e}")
        return {"history": []}
    finally:
        session.close()
