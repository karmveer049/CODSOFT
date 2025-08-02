
# 📊 Customer Churn Prediction

This project predicts whether a customer will churn (i.e., stop using a subscription-based service) based on historical customer data, including demographic and usage behavior.

## 🧠 Algorithms Used

Three machine learning models were tested:
- Logistic Regression
- Random Forest ✅ *(Best performing model based on F1-score)*
- Gradient Boosting

The final model selected was **Random Forest** due to its superior performance on evaluation metrics.

---

## 📁 Dataset

The dataset used is:
`Churn_Modelling.csv`

It includes the following features:
- CreditScore
- Geography (France/Germany/Spain)
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card (1/0)
- Is Active Member (1/0)
- Estimated Salary
- Exited (Target: 1 = Churned, 0 = Stayed)

---

## ⚙️ How It Works

### 1. Train the Model

The script `train_churn_model.py`:
- Loads and preprocesses the dataset.
- Encodes categorical features (`Geography`, `Gender`).
- Scales numerical features.
- Trains 3 models and selects the best using F1-score.
- Saves the final model to `churn_model.pkl` and the scaler to `scaler.pkl`.

### 2. Predict on New Input

The script `predict_churn.py`:
- Loads the saved model and scaler.
- Takes user input from the command line.
- Preprocesses and scales the input.
- Predicts whether the customer will churn or not.

---

## 💾 Files

- `train_churn_model.py` – Trains and saves the best model
- `predict_churn.py` – Runs churn prediction on new input
- `churn_model.pkl` – Trained Random Forest model (binary format)
- `scaler.pkl` – Scaler used during preprocessing
- `Churn_Modelling.csv` – Customer data for training

---

## 🖥️ How to Run

### Step 1: Train the model
```bash
python train_churn_model.py
````

### Step 2: Make predictions

```bash
python predict_churn.py
```

---

## 📌 Requirements

* Python 3.x
* pandas
* numpy
* scikit-learn

Install dependencies:

```bash
pip install pandas numpy scikit-learn
```

---

## 🧪 Example Prediction Flow

```plaintext
Enter Customer Details:
Credit Score: 600
Geography (France/Germany/Spain): France
Gender (Male/Female): Female
Age: 42
Tenure: 2
Balance: 100000
Number of Products: 1
Has Credit Card (1/0): 1
Is Active Member (1/0): 1
Estimated Salary: 90000

🔍 Prediction: Customer will stay.
```

---

## ✍️ Author

Developed by **Karmveer** using Python and scikit-learn for model training and prediction.


