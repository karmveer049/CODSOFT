import joblib
import re

# === File paths (local system paths) ===
model_path = r"C:\Users\karam\OneDrive\Desktop\CodSoft\Spam\spam_model_tfidf.pkl"
vectorizer_path = r"C:\Users\karam\OneDrive\Desktop\CodSoft\Spam\spam_vectorizer.pkl"

# === Load trained model and vectorizer ===
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# === Clean input text (same as during training) ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\b\w{1,2}\b', '', text)
    return text.strip()

# === Prediction loop ===
while True:
    user_input = input("\nEnter a message to check for spam (or type 'exit' to quit):\n> ")
    if user_input.lower() == 'exit':
        print("👋 Exiting spam checker.")
        break

    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    if prediction == 1:
        print("🚫 This message is likely **SPAM**.")
    else:
        print("✅ This message is likely **NOT spam**.")
