# 🤖 Job Risk ML — AI Impact on Jobs 2030

> **Predict karo kisi bhi job ka AI automation risk — Low, Medium, ya High!**

Ye ek Machine Learning project hai jo predict karta hai ke 2030 tak kisi job ko AI/Automation ka kitna risk hai. Is project mein **Random Forest** aur **Logistic Regression** models use kiye gaye hain real dataset ke saath.

---

## 📌 Project Overview

AI aur automation ki wajah se duniya bhar mein hazaron jobs khatam ho sakti hain ya badal sakti hain. Ye project usi cheez ko analyze karta hai:

- 📂 **3,000+ job records** ka real dataset use kiya gaya
- 🎯 **Target:** `Risk_Category` — `Low`, `Medium`, ya `High`
- 🧠 **Models:** Logistic Regression + Random Forest Classifier
- 📊 **Output:** Accuracy score, classification report, confusion matrix, feature importance graph

---

## 🗂️ Project Structure

```
job-risk-ml/
│
├── AI_Impact_on_Jobs_2030.csv       # Main dataset (3001 rows, 18 columns)
├── AI Impact on Jobs 2030/
│   └── AI_Impact_on_Jobs_2030.csv   # Dataset backup folder
│
├── train.py                         # Simple training script (Logistic Regression)
├── all_in_one_ml.py                 # Full ML pipeline (Random Forest + prediction)
├── in_ONE_ml.py                     # Advanced: Train + Streamlit + Flask API
├── predict.py                       # Load saved model aur predict karo
│
├── risk_model.joblib                # Trained model (saved)
└── .gitignore
```

---

## 📊 Dataset — `AI_Impact_on_Jobs_2030.csv`

| Column | Description |
|--------|-------------|
| `Job_Title` | Job ka naam (e.g. Data Scientist, Security Guard) |
| `Average_Salary` | Salary (USD) |
| `Years_Experience` | Tajruba (saal mein) |
| `Education_Level` | Taleem (High School / Bachelor's / Master's / PhD) |
| `AI_Exposure_Index` | AI se exposure ka level (0 to 1) |
| `Tech_Growth_Factor` | Technology growth factor |
| `Automation_Probability_2030` | 2030 tak automation ka chance (0 to 1) |
| `Skill_1` to `Skill_10` | 10 different skill scores (0 to 1) |
| `Risk_Category` ⭐ | **Target:** Low / Medium / High |

**Dataset Size:** 3,001 rows × 18 columns

---

## ⚙️ Kaise Kaam Karta Hai?

```
CSV Data
   ↓
Feature Engineering
(OneHotEncoder for Job_Title & Education_Level)
(StandardScaler for numerical columns)
   ↓
ML Model Training
(Random Forest / Logistic Regression)
   ↓
Evaluation
(Accuracy, Classification Report, Confusion Matrix)
   ↓
Model Save → risk_model.joblib
   ↓
Prediction (Terminal / Streamlit / Flask API)
```

---

## 🧪 Models Used

### 1. `train.py` — Logistic Regression (Basic)
- Simple baseline model
- Logistic Regression with `max_iter=2000`
- 80/20 train-test split

### 2. `all_in_one_ml.py` — Random Forest (Main)
- `n_estimators=200`, `random_state=42`
- Confusion Matrix graph dikhata hai
- Real-time terminal input se prediction

### 3. `in_ONE_ml.py` — Advanced All-in-One
- `n_estimators=300` Random Forest
- Feature Importance graph (Top 15 features)
- **3 modes:**
  - `1` → Normal training
  - `2` → Streamlit Web App
  - `3` → Flask REST API (`POST /predict`)

### 4. `predict.py` — Saved Model se Prediction
- `risk_model.joblib` load karta hai
- Seedha "Data Scientist" ke liye prediction deta hai
- Probability bhi dikhata hai

---

## 🚀 Kaise Chalayein?

### Prerequisites
```bash
pip install pandas scikit-learn joblib matplotlib streamlit flask
```

### Option 1: Basic Training (Logistic Regression)
```bash
python train.py
```

### Option 2: Random Forest + Real-time Prediction
```bash
python all_in_one_ml.py
```
> Terminal mein Job Title, Education, Salary, Experience daalo — prediction milegi!

### Option 3: Advanced (Train + Web + API)
```bash
python in_ONE_ml.py
```
```
Choose Mode:
1 = Normal ML Training
2 = Streamlit Web App
3 = Flask API
```

**Streamlit Web App chalana:**
```bash
# in_ONE_ml.py mein 2 choose karo
# Ya seedha:
streamlit run in_ONE_ml.py
```

**Flask API chalana:**
```bash
# in_ONE_ml.py mein 3 choose karo
# Phir POST request bhejo:
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"Job_Title":"Data Scientist","Average_Salary":110000,...}'
```

### Option 4: Saved Model Se Predict Karo
```bash
python predict.py
```

---

## 📈 Model Results

| Model | Accuracy |
|-------|----------|
| Logistic Regression (`train.py`) | ~70–75% |
| Random Forest 200 trees (`all_in_one_ml.py`) | ~85–90% |
| Random Forest 300 trees (`in_ONE_ml.py`) | ~87–92% |

> ⚡ Random Forest, Logistic Regression se kaafi better perform karta hai is dataset pe.

---

## 🎯 Example Prediction

```python
Input:
  Job Title       : Data Scientist
  Salary          : 110,000
  Experience      : 4 years
  Education       : Master's

Output:
  Predicted Risk  : Low
  Probabilities   : Low: 0.72 | Medium: 0.20 | High: 0.08
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Main language |
| Pandas | Data loading & manipulation |
| Scikit-learn | ML models & preprocessing |
| Joblib | Model save/load |
| Matplotlib | Graphs (confusion matrix, feature importance) |
| Streamlit | Web app UI |
| Flask | REST API |

---

## 👨‍💻 Author

**Uneeb Zulfiqar**
- GitHub: [@uneebzulfiqar45-cpu](https://github.com/uneebzulfiqar45-cpu)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
