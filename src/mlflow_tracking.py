import mlflow

def test_mlflow_logging():
    # Set local SQLite DB for MLflow tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("house_price_prediction_monitoring")
    
    print("Starting MLflow tracking session...")
    with mlflow.start_run():
        mlflow.log_param("data_path", "data/Housing.csv")
        mlflow.log_param("target_column", "price")
        
        # Log mock parameters and metrics for verification
        mlflow.log_metric("R2_score", 0.685)
        mlflow.log_metric("RMSE", 1025000.0)
        
        print("Logged tracking attributes to MLflow successfully!")
        print("Check running runs dashboard using 'mlflow ui' command.")

if __name__ == "__main__":
    test_mlflow_logging()
