import joblib

# ------------------ Load Saved Model and Label Encoder ------------------
model_path = r"C:\Users\karam\OneDrive\Desktop\CodSoft\MovieGenrePridiction\genre_model.pkl"
encoder_path = r"C:\Users\karam\OneDrive\Desktop\CodSoft\MovieGenrePridiction\label_encoder.pkl"

print("Loading saved model and label encoder...")
model = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)
print("✅ Model and encoder loaded successfully!")

# ------------------ Prediction Loop ------------------
while True:
    print("\n--- Predict Movie Genre ---")
    movie_id = input("Enter Movie ID: ")
    title = input("Enter Movie Title: ")
    description = input("Enter Movie Description: ")

    # Combine title and description as input
    input_text = title + " " + description

    # Predict genre
    prediction = model.predict([input_text])[0]
    predicted_genre = label_encoder.inverse_transform([prediction])[0]

    print(f"\n🎬 Predicted Genre for Movie ID {movie_id}: {predicted_genre.strip()}")

    again = input("\nDo you want to predict another movie? (yes/no): ").strip().lower()
    if again != 'yes':
        print("\n👋 Exiting prediction loop. Goodbye!")
        break
