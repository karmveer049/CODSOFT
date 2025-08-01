🎬 Movie Genre Prediction using Logistic Regression
This project aims to predict the genre of a movie based on its title and description using a Logistic Regression model trained on a custom dataset.

📁 Dataset
Format: CSV

File Name: train_data.csv

Columns:

id: Movie ID

title: Movie title (e.g., "Inception")

genre: Genre label (e.g., "Action")

description: Movie description (e.g., "A thief who steals corporate secrets...")



🧠 Model
Algorithm: Logistic Regression

Vectorization: TF-IDF (Term Frequency–Inverse Document Frequency)

Pipeline: TF-IDF → Logistic Regression

📊 Accuracy
Average accuracy: ~58-60%

Depends on dataset balance and text quality
🧠 Model Persistence
After training the movie genre prediction model using Logistic Regression and TF-IDF vectorization, the model was saved as a serialized .pkl file for reuse without retraining.

✅ What’s Included:
genre_model.pkl — Pickled model pipeline (TF-IDF + Logistic Regression)

label_encoder.pkl — Pickled label encoder to map predicted class indices to genre names

runPicklefile.py — Python script to load the .pkl files and make real-time genre predictions from user input
