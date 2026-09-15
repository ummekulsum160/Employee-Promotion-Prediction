import os
import sys
from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "employee_promotion_model.pkl"
model = joblib.load(MODEL_PATH)

employee = pd.DataFrame([{
    "age": 31,
    "years_at_company": 5,
    "years_in_current_role": 2,
    "performance_score": 4,
    "training_hours": 45,
    "education_level": 3,
    "previous_promotions": 1,
    "job_satisfaction": 4
}])

prediction = model.predict(employee)[0]
probability = model.predict_proba(employee)[0][1]

print("Promotion Prediction:", "Promoted" if prediction == 1 else "Not Promoted")
print(f"Promotion probability: {probability:.2%}")
