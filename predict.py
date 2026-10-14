import pandas as pd
import joblib

model = joblib.load("risk_model.joblib")

sample = pd.DataFrame([{
    "Job_Title": "Data Scientist",
    "Average_Salary": 110000,
    "Years_Experience": 4,
    "Education_Level": "Master's",
    "AI_Exposure_Index": 0.6,
    "Tech_Growth_Factor": 1.2,
    "Automation_Probability_2030": 0.4,
    "Skill_1": 0.7,
    "Skill_2": 0.6,
    "Skill_3": 0.5,
    "Skill_4": 0.4,
    "Skill_5": 0.8,
    "Skill_6": 0.6,
    "Skill_7": 0.3,
    "Skill_8": 0.9,
    "Skill_9": 0.5,
    "Skill_10": 0.4
}])

pred = model.predict(sample)[0]
prob = model.predict_proba(sample)[0]

print("Predicted Risk:", pred)
print("Probabilities:", prob)
