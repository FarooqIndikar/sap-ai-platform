import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from etl.preprocess import load_data, clean_data, feature_engineering
import joblib

df = load_data("data/sap_data.csv")
df = clean_data(df)
df = feature_engineering(df)

X = df[["amount", "due_days", "risk_score"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(classification_report(y_test, predictions))

joblib.dump(model, "ml/model.pkl")
print("Model saved successfully.")