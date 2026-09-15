import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = "data/employee_promotion.csv"
MODEL_PATH = "models/employee_promotion_model.pkl"

df = pd.read_csv(DATA_PATH)

features = [
    "age",
    "years_at_company",
    "years_in_current_role",
    "performance_score",
    "training_hours",
    "education_level",
    "previous_promotions",
    "job_satisfaction"
]
target = "promoted"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Employee Promotion Prediction Results")
print("-" * 42)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")

importance = pd.Series(
    model.feature_importances_, index=features
).sort_values()

importance.plot(kind="barh", title="Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/feature_importance.png")
plt.close()
print("Chart saved to: outputs/feature_importance.png")
