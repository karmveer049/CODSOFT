import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

try:
    # Load the dataset
    file_path = r"C:\Users\karam\OneDrive\Desktop\CodSoft\Churn_Modelling.csv"
    df = pd.read_csv(file_path)
    print("✅ CSV file loaded successfully.")

    # Drop unnecessary columns
    df_model = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

    # Encode categorical variables
    df_model['Gender'] = LabelEncoder().fit_transform(df_model['Gender'])
    df_model['Geography'] = LabelEncoder().fit_transform(df_model['Geography'])

    # Features and target
    X = df_model.drop('Exited', axis=1)
    y = df_model['Exited']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train multiple models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'model': model
        }
        print(f"{name} - Accuracy: {results[name]['accuracy']:.4f}, F1 Score: {results[name]['f1_score']:.4f}")

    # Choose best model by F1-score
    best_model_name = max(results, key=lambda x: results[x]['f1_score'])
    best_model = results[best_model_name]['model']
    print(f"✅ Best model selected: {best_model_name}")

    # Save model and scaler to the same directory
    model_path = os.path.join(os.path.dirname(file_path), "churn_model.pkl")
    scaler_path = os.path.join(os.path.dirname(file_path), "scaler.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Scaler saved to: {scaler_path}")

except Exception as e:
    print("❌ Error occurred:", e)
