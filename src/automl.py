import pandas as pd
import os

def run_automl():
    data_path = os.path.join("data", "Housing.csv")
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    print(f"Dataset Loaded. Running AutoML regressor comparison on {len(df)} samples...")
    
    try:
        # PyCaret 4.x Object Oriented API
        from pycaret.regression import RegressionExperiment
        exp = RegressionExperiment(session_id=123, verbose=False)
        exp.fit(X=df.drop("price", axis=1), y=df["price"])
        best_model = exp.compare_models()
        print("\nAutoML Execution Completed successfully!")
        print("Best Fitting Estimator:", best_model.best)
    except ImportError:
        # PyCaret 3.x Functional API Fallback
        from pycaret.regression import setup, compare_models
        setup(data=df, target="price", verbose=False, html=False, session_id=123)
        best_model = compare_models()
        print("\nAutoML Execution Completed successfully (via legacy API)!")
        print("Best Fitting Estimator:", best_model)

if __name__ == "__main__":
    run_automl()
