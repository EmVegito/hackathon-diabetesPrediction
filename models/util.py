import joblib

def save_model(model, filename="fraud_detection_model.joblib"):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")