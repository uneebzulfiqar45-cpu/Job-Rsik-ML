# =========================================================
# ALL IN ONE ML PROJECT (TRAIN + GRAPH + WEB + API)
# =========================================================

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# -------------------------
# CONFIG
# -------------------------
CSV_PATH = "AI_Impact_on_Jobs_2030.csv"
MODEL_PATH = "risk_model.joblib"

# -------------------------
# 1. LOAD DATA
# -------------------------
df = pd.read_csv(CSV_PATH)

y = df["Risk_Category"]
X = df.drop(columns=["Risk_Category"])

cat_cols = ["Job_Title", "Education_Level"]
num_cols = [c for c in X.columns if c not in cat_cols]

# -------------------------
# 2. PREPROCESSING
# -------------------------
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols)
    ]
)

# -------------------------
# 3. MODEL
# -------------------------
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

pipeline = Pipeline([
    ("prep", preprocess),
    ("model", model)
])

# -------------------------
# 4. TRAIN / TEST
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)

# -------------------------
# 5. EVALUATION
# -------------------------
y_pred = pipeline.predict(X_test)

print("\n========= MODEL RESULT =========")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -------------------------
# 6. CONFUSION MATRIX
# -------------------------
cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
disp = ConfusionMatrixDisplay(cm, display_labels=pipeline.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# -------------------------
# 7. FEATURE IMPORTANCE
# -------------------------
feature_names = (
    pipeline.named_steps["prep"]
    .get_feature_names_out()
)

importances = pipeline.named_steps["model"].feature_importances_

fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)[:15]

plt.figure(figsize=(8, 5))
fi.plot(kind="barh")
plt.title("Top 15 Feature Importances")
plt.gca().invert_yaxis()
plt.show()

# -------------------------
# 8. SAVE MODEL
# -------------------------
joblib.dump(pipeline, MODEL_PATH)
print("Model saved:", MODEL_PATH)

# =========================================================
# 9. STREAMLIT WEB APP
# =========================================================
def run_streamlit():
    import streamlit as st

    st.title("Job Risk Prediction (AI Impact 2030)")

    job = st.text_input("Job Title", "Data Scientist")
    edu = st.selectbox("Education Level", ["Bachelor's", "Master's", "PhD"])
    salary = st.number_input("Average Salary", 30000, 200000, 100000)
    exp = st.slider("Years Experience", 0, 20, 5)

    if st.button("Predict Risk"):
        sample = pd.DataFrame([{
            "Job_Title": job,
            "Average_Salary": salary,
            "Years_Experience": exp,
            "Education_Level": edu,
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

        model = joblib.load(MODEL_PATH)
        pred = model.predict(sample)[0]
        st.success(f"Predicted Risk: {pred}")

# =========================================================
# 10. FLASK API
# =========================================================
def run_flask():
    from flask import Flask, request, jsonify

    app = Flask(__name__)
    model = joblib.load(MODEL_PATH)

    @app.route("/predict", methods=["POST"])
    def predict():
        data = request.json
        df = pd.DataFrame([data])
        pred = model.predict(df)[0]
        return jsonify({"Risk_Category": pred})

    app.run(debug=True)

# =========================================================
# 11. RUN MODE
# =========================================================
print("""
Choose Mode:
1 = Normal ML Training (already done)
2 = Streamlit Web App
3 = Flask API
""")

choice = input("Enter choice (1/2/3): ")

if choice == "2":
    run_streamlit()
elif choice == "3":
    run_flask()
else:
    print("Training completed.")
