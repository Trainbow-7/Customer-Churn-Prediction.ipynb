# Customer Churn Prediction 🏦

Predicts whether a bank customer will churn (leave the bank) using historical customer data such as credit score, age, balance, geography, and account activity. Built with a Random Forest classifier and a Keras neural network, achieving ~86% accuracy — helping identify at-risk customers so businesses can act early to retain them.

🔗 **Live Demo:** [customer-churn-prediction-ipynb.onrender.com](https://customer-churn-prediction-ipynb.onrender.com)
> Note: hosted on Render's free tier — the app may take 30–60 seconds to wake up if it's been inactive.

## 📊 Dataset
`Churn_Modelling.csv` — 10,000 bank customer records with features like credit score, age, tenure, balance, number of products, credit card status, activity status, estimated salary, and geography.

## 🚀 Features
- Data preprocessing & encoding (Label + One-Hot Encoding)
- Feature scaling with StandardScaler
- Random Forest Classifier (baseline model)
- Artificial Neural Network (Keras/TensorFlow)
- Model evaluation (accuracy, confusion matrix, classification report)
- Feature importance analysis
- Streamlit web app with worldwide country support for live predictions

## 📁 Repository Structure

```
├── Customer_Churn_Prediction.ipynb   # Full training notebook (EDA, preprocessing, training, evaluation)
├── app.py                            # Streamlit prediction app
├── churn_model.pkl                   # Trained Random Forest model
├── churn_ann_model.h5                # Trained ANN model
├── scaler.pkl                        # StandardScaler used for feature scaling
├── gender_encoder.pkl                # LabelEncoder for the Gender column
├── Churn_Modelling.csv               # Dataset
├── requirements.txt                  # Python dependencies
├── runtime.txt                       # Pinned Python version
└── README.md
```

## ⚙️ Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the app locally:
```bash
streamlit run app.py
```

Or load the model directly in Python:
```python
import joblib
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
```

## 📈 Results
| Model | Accuracy |
|---|---|
| Random Forest | ~86% |
| Neural Network (ANN) | ~86.5% |

## 🛠️ Tech Stack
Python · Pandas · Scikit-learn · TensorFlow/Keras · Streamlit · Render (deployment)
