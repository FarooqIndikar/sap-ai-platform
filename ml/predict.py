import joblib
import pandas as pd

def predict_new(data_dict):
    model = joblib.load("ml/model.pkl")
    df = pd.DataFrame([data_dict])
    prediction = model.predict(df)
    return int(prediction[0])


if __name__ == "__main__":
    sample = {
        "amount": 150000,
        "due_days": 40,
        "risk_score": 150000 * 0.6 + 40 * 0.4
    }

    result = predict_new(sample)
    print("Prediction:", result)