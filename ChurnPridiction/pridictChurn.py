import pickle
import numpy as np

# Load saved model and scaler
with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Take user input
def get_user_input():
    print("Enter Customer Details:")
    CreditScore = int(input("Credit Score: "))
    Geography = input("Geography (France/Germany/Spain): ").strip().lower()
    Gender = input("Gender (Male/Female): ").strip().lower()
    Age = int(input("Age: "))
    Tenure = int(input("Tenure: "))
    Balance = float(input("Balance: "))
    NumOfProducts = int(input("Number of Products: "))
    HasCrCard = int(input("Has Credit Card (1/0): "))
    IsActiveMember = int(input("Is Active Member (1/0): "))
    EstimatedSalary = float(input("Estimated Salary: "))

    geo_map = {'france': 0, 'germany': 1, 'spain': 2}
    gender_map = {'female': 0, 'male': 1}

    Geography = geo_map.get(Geography, 0)
    Gender = gender_map.get(Gender, 0)

    return np.array([[CreditScore, Geography, Gender, Age, Tenure,
                      Balance, NumOfProducts, HasCrCard, IsActiveMember,
                      EstimatedSalary]])

# Predict churn
user_input = get_user_input()
user_input_scaled = scaler.transform(user_input)
prediction = model.predict(user_input_scaled)

print("\n🔍 Prediction:", "Customer will churn." if prediction[0] == 1 else "Customer will stay.")
