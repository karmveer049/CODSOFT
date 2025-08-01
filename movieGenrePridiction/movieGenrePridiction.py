import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# Load dataset
file_path = r"C:\Users\karam\OneDrive\Desktop\CodSoft\MovieGenrePridiction\train_data.csv"
df = pd.read_csv(file_path, names=['id', 'title', 'genre', 'description'], header=None)



# Combine title and description
df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')

# Encode the genres
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['genre'].str.strip())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.2, random_state=42)

# Build the model with TF-IDF limit for speed
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train the model
print("Training model... Please wait.")
model.fit(X_train, y_train)
print(f"\n✅ Model trained successfully. Test Accuracy: {model.score(X_test, y_test):.2f}\n")

# Loop for multiple predictions
while True:
    print("\n--- Predict Movie Genre ---")
    movie_id = input("Enter Movie ID: ")
    title = input("Enter Movie Title: ")
    description = input("Enter Movie Description: ")

    input_text = title + " " + description
    prediction = model.predict([input_text])[0]
    predicted_genre = label_encoder.inverse_transform([prediction])[0]

    print(f"\n🎬 Predicted Genre for Movie ID {movie_id}: {predicted_genre.strip()}")

    again = input("\nDo you want to predict another movie? (yes/no): ").strip().lower()
    if again != 'yes':
        print("\n👋 Exiting prediction loop. Goodbye!")
        break
