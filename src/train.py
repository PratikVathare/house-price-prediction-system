import pandas as pd
import mlflow
import os

def train_model():
    print("Initializing Model Training Pipeline...")
    
    # 1. Load data
    data_path = os.path.join("data", "Housing.csv")
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    print(f"Loaded dataset successfully with shape: {df.shape}")
    
    # 2. Configure MLflow tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("house_price_prediction_system")
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    model_save_path = os.path.join("models", "house_price")
    
    # 3. Setup PyCaret and Train Best Regressor
    try:
        # PyCaret 4.x Object Oriented API
        from pycaret.regression import RegressionExperiment
        exp = RegressionExperiment(
            session_id=123,
            train_size=0.8,
            fold=5,
            log_experiment=True,
            verbose=False
        )
        exp.fit(
            X=df.drop("price", axis=1),
            y=df["price"]
        )
        print("Running model comparison search...")
        comparison_result = exp.compare_models()
        print(f"Training completed. Best Model Selected: {comparison_result.best}")
        
        print("Finalizing model...")
        final_model = exp.finalize_model(comparison_result.best)
        
        # In PyCaret 4.x, finalize_model returns a FinalizeResult object. Extract the actual Pipeline.
        pipeline_to_save = final_model.pipeline if hasattr(final_model, "pipeline") else final_model
        
        print(f"Saving serialized model pipeline to '{model_save_path}.pkl'...")
        exp.save_model(pipeline_to_save, model_save_path)
        
    except ImportError:
        # PyCaret 3.x Functional API Fallback
        from pycaret.regression import setup, compare_models, finalize_model, save_model
        setup(
            data=df,
            target="price",
            train_size=0.8,
            session_id=123,
            log_experiment=True,
            experiment_name="house_price_prediction_system",
            log_data=True,
            verbose=False,
            html=False
        )
        print("Running model comparison search...")
        best_model = compare_models(fold=5)
        print(f"Training completed. Best Model Selected: {best_model}")
        
        print("Finalizing model...")
        final_model = finalize_model(best_model)
        
        print(f"Saving serialized model pipeline to '{model_save_path}.pkl'...")
        save_model(final_model, model_save_path)
        
    print("\nTraining completed successfully! Saved model pipeline to models/house_price.pkl")

if __name__ == "__main__":
    train_model()
