# 🧠 Titanic Survival Prediction – Machine Learning Project

## 📌 Overview
**This project aims to predict passenger survival on the Titanic using various machine learning classification models.**  
**The goal is to compare model performance and identify the most accurate predictor based on the Titanic dataset from [Kaggle](https://www.kaggle.com/c/titanic).**

---

## 🎯 Objectives
- **Train and evaluate multiple classification models on the Titanic dataset.**
- **Identify the best-performing model for survival prediction.**
- **Prepare the final model for deployment.**

---

## 🛠️ Tech Stack & Tools
- **Programming Language:** Python  
- **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`  
- **Modeling Techniques:** Logistic Regression, Support Vector Machine, KNN, Decision Tree  
- **Preprocessing:** OneHotEncoding, StandardScaler  
- **Visualization:** Matplotlib, Seaborn, `plot_tree()` from sklearn  

---

## 🧱 Project Structure
```
titanic-sink-analysis/
│
├── saved_models/                 # Trained model files (SVC, scaler, column names)
│   ├── svc_model.pkl
│   ├── scaler.pkl
│   └── columns.pkl
│
├── Titanic-Dataset.csv          # Dataset used for training and testing
├── Titanic_Sink.ipynb           # Main notebook for training and evaluation
├── requirements.txt             # List of required libraries
└── README.md                    # Project summary
```

---

## 📊 Data Preprocessing
- **Selected features:** `Pclass`, `Age`, `SibSp`, `Fare`, `Sex`, `Embarked`  
- **Handled missing values**  
- **Encoded categorical variables using OneHotEncoding**  
- **Scaled numerical features using StandardScaler**  

---

## 🤖 Models Trained

| **Model**                   | **Accuracy** | **Cross-Validation** | **Notes**                                                   |
|----------------------------|--------------|----------------------|--------------------------------------------------------------|
| **Logistic Regression**    | **~82.3%**   | **~82.3%**           | **Balanced performance**                                     |
| **Support Vector Classifier** | **~82.9%** | **~82.3%**           | **Best performer with default RBF kernel**                   |
| **KNN (n=5)**              | **~80.6%**   | **~82.2%**           | **Performed well with default settings**                     |
| **Decision Tree**          | **~78.9%**   | **~79.0%**           | **Performance dropped after tuning**                         |

---

## 📈 Visualizations
- **Bar chart of model accuracies**  
- **Decision tree visualization using `plot_tree()`**  

---

## ✅ Final Outcome
- **Best Model:** Support Vector Classifier (default settings)  
- **Saved using:** `joblib` (includes scaler and column metadata)  
- **Ready for:** Deployment or extension with ensemble models (e.g., Random Forest, XGBoost)  

---

## 🚀 How to Run

### 1. Clone the repo:
```bash
git clone https://github.com/your-username/titanic-sink-analysis.git
cd titanic-sink-analysis
```

### 2. Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook:
```bash
jupyter notebook Titanic_Sink.ipynb
```

---

## 📂 Future Work
- **Add ensemble methods (e.g., Random Forest, XGBoost)**  
- **Deploy as a web app using Flask or Streamlit**  
- **Feature engineering and hyperparameter tuning**  

---

## 🧑‍💻 Author
**Nayan Darokar**
