import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. CSV load
df = pd.read_csv("AI_Impact_on_Jobs_2030.csv")

# 2. Target & features
y = df["Risk_Category"]
X = df.drop(columns=["Risk_Category"])

# 3. Columns
cat_cols = ["Job_Title", "Education_Level"]
num_cols = [c for c in X.columns if c not in cat_cols]

# 4. Preprocessing
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols)
    ]
)

# 5. Model
model = LogisticRegression(max_iter=2000)

# 6. Pipeline
pipeline = Pipeline([
    ("prep", preprocess),
    ("model", model)
])

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Train
pipeline.fit(X_train, y_train)

# 9. Evaluate
pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# 10. Save model
joblib.dump(pipeline, "risk_model.joblib")
print("Model saved as risk_model.joblib")
