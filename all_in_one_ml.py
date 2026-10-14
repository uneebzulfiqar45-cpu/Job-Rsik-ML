import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# ===============================
# 1. DATA LOAD
# ===============================
df = pd.read_csv("AI_Impact_on_Jobs_2030.csv")

# Target & Features
y = df["Risk_Category"]
X = df.drop(columns=["Risk_Category"])

# Categorical & Numerical columns
cat_cols = ["Job_Title", "Education_Level"]
num_cols = [c for c in X.columns if c not in cat_cols]

# ===============================
# 2. PREPROCESSING
# ===============================
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols)
    ]
)

# ===============================
# 3. MODEL (Random Forest)
# ===============================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

pipeline = Pipeline([
    ("prep", preprocess),
    ("model", model)
])

# ===============================
# 4. TRAIN / TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# 5. TRAIN MODEL
# ===============================
pipeline.fit(X_train, y_train)

# ===============================
# 6. EVALUATION
# ===============================
y_pred = pipeline.predict(X_test)

print("\n================ RESULT ================")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ===============================
# 7. CONFUSION MATRIX GRAPH
# ===============================
cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# ===============================
# 8. SAVE MODEL
# ===============================
joblib.dump(pipeline, "risk_model.joblib")
print("\nModel saved as risk_model.joblib")

# ===============================
# 9. REAL-TIME USER INPUT
# ===============================
print("\n========= REAL-TIME PREDICTION =========")

job_title = input("Job Title: ")
education = input("Education Level (Bachelor's/Master's/etc): ")
salary = float(input("Average Salary: "))
experience = float(input("Years Experience: "))

sample = pd.DataFrame([{
    "Job_Title": job_title,
    "Average_Salary": salary,
    "Years_Experience": experience,
    "Education_Level": education,
    "AI_Exposure_Index": 0.5,
    "Tech_Growth_Factor": 1.0,
    "Automation_Probability_2030": 0.5,
    "Skill_1": 0.5,
    "Skill_2": 0.5,
    "Skill_3": 0.5,
    "Skill_4": 0.5,
    "Skill_5": 0.5,
    "Skill_6": 0.5,
    "Skill_7": 0.5,
    "Skill_8": 0.5,
    "Skill_9": 0.5,
    "Skill_10": 0.5
}])

prediction = pipeline.predict(sample)[0]
probabilities = pipeline.predict_proba(sample)[0]

print("\nPredicted Risk Category:", prediction)
print("Probabilities:")
for cls, prob in zip(pipeline.classes_, probabilities):
    print(f"{cls}: {prob:.2f}")
