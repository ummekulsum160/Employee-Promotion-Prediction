# How to Run: Employee Promotion Prediction using Scikit-learn

This project predicts whether an employee may be promoted based on performance, training, education, experience, and previous promotion history.

---

## 1. Prerequisites

- **Python**: Python 3.9+ (Python 3.14 tested)
- **Terminal**: Windows PowerShell, Command Prompt, or VS Code integrated terminal
- **Dependencies**: `pandas`, `scikit-learn`, `numpy`, `joblib`, `matplotlib`

---

## 2. Open Terminal & Navigate to Project Directory

Open PowerShell or Command Prompt, then change directory to the project folder:

```powershell
cd "c:\Users\user\Desktop\jj college\Employee_Promotion_Prediction_Sklearn"
```

---

## 3. Install Dependencies

Install required packages:

```powershell
pip install -r requirements.txt
```

> **Windows Note**: If `python` or `pip` opens the Microsoft Store, specify the direct Python path:
> ```powershell
> & "C:\Users\user\AppData\Local\Python\bin\python.exe" -m pip install -r requirements.txt
> ```

---

## 4. Step 1: Train the Machine Learning Model

Execute the training script to train and save the model:

```powershell
python train_model.py
```
*(or `py train_model.py` / `& "C:\Users\user\AppData\Local\Python\bin\python.exe" train_model.py`)*

### What this does:
1. Loads the dataset from `data/employee_promotion.csv`.
2. Trains a **RandomForestClassifier** model predicting `promoted`.
3. Evaluates model performance (Accuracy Score, Classification Report).
4. Saves the trained model (`.pkl` file).
5. Generates and saves visualization plots/charts (e.g., feature importance or segment visualizations).

---

## 5. Step 2: Run Predictions

Run the prediction script to test predictions:

```powershell
python predict.py
```

### Sample Prediction:
The script runs a sample test case directly from preconfigured values:

```python
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
```

To customize input values, edit [predict.py](predict.py) directly and run `python predict.py` again.

---

## 6. Directory Structure

```text
Employee_Promotion_Prediction_Sklearn/
├── data/
│   └── employee_promotion.csv
├── predict.py
├── train_model.py
├── requirements.txt
├── README.md
└── HOW_TO_RUN.md
```
