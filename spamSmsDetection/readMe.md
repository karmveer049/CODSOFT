
# 📧 Spam Message Detection using Machine Learning

This project is a simple yet effective spam message detector built using **TF-IDF vectorization** and a **Machine Learning classifier** (e.g., Multinomial Naive Bayes or Logistic Regression). It classifies user-inputted messages as either **Spam** or **Not Spam**.

---

## 📁 Project Structure

```

Spam/
├── spam\_model\_tfidf.pkl         # Trained ML model
├── spam\_vectorizer.pkl          # TF-IDF vectorizer used during training
├── spam\_checker.py              # Script to input messages and check for spam
├── README.md                    # Project documentation (this file)
├── spam.csv                     # Original dataset (optional for retraining)

````

---

## 🛠 Requirements

- Python 3.x
- Libraries:
  - `scikit-learn`
  - `joblib`
  - `pandas` (if retraining)
  - `re` (built-in)

Install dependencies (if needed):

```bash
pip install scikit-learn pandas
````

---

## 🚀 How to Run

1. Ensure you have `spam_model_tfidf.pkl` and `spam_vectorizer.pkl` in the same directory.
2. Run the script:

```bash
python spam_checker.py
```

3. Input your message in the terminal when prompted.
4. Type `'exit'` to quit.

---

## 💡 How It Works

* **Preprocessing**: Removes special characters, extra spaces, and short words.
* **TF-IDF Vectorization**: Converts text into numerical features based on term frequency and inverse document frequency.
* **Model Prediction**: The trained model predicts if the input message is spam (`1`) or not (`0`).

---





## 🧠 Model Accuracy

During training (optional step), you can evaluate the model using `accuracy_score`, `confusion_matrix`, and `classification_report` from `sklearn.metrics`.

---

## 👨‍💻 Author

Developed by **Karamveer Kumar** as part of a project under **CodSoft Internship**.

---

## 📜 License

This project is open-source and free to use for educational purposes.


