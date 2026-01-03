# 📧 Spam Email (SMS) Classifier using Machine Learning

## 📌 Project Overview

This project is a **Spam Email (SMS) Classification system** built using **Machine Learning and Natural Language Processing (NLP)** techniques.
The model predicts whether a given SMS/message is **Spam** or **Not Spam (Ham)**.

The goal of this project is to demonstrate a **complete ML workflow**, including text preprocessing, feature extraction, model training, and evaluation on an imbalanced dataset.

---

## 🎯 Problem Type

- **Supervised Learning**
- **Binary Classification**
- **Text Classification (NLP)**

---

## 📊 Dataset

- **Dataset Name:** SMS Spam Collection (UCI Machine Learning Repository)
- **Total Messages:** 5,572
- **Class Distribution:**

  - Ham (Not Spam): 4,825
  - Spam: 747

This dataset is **naturally imbalanced**, reflecting real-world email/SMS data.

---

## 🛠️ Technologies & Tools

- Python
- NumPy
- Pandas
- Scikit-learn
- NLTK
- Matplotlib / Seaborn
- VS Code & Jupyter Notebook

---

## 🔄 Machine Learning Workflow

### 1️⃣ Data Preprocessing

- Converted text to lowercase
- Removed punctuation and special characters
- Removed stopwords
- Tokenization and lemmatization

### 2️⃣ Feature Extraction

- **TF-IDF Vectorization**
- Converted text data into numerical features

### 3️⃣ Data Splitting

- **Stratified Train-Test Split (80/20)**
- Ensured equal spam/ham ratio in both sets

### 4️⃣ Model Used

- **Multinomial Naive Bayes**
- Well-suited for text classification problems

---

## 📈 Model Evaluation

### 🔹 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

### 🔹 Results

```
Accuracy: 0.9713

              precision    recall  f1-score   support

Ham (0)         0.97       1.00      0.98       966
Spam (1)        0.99       0.79      0.88       149

Overall Accuracy: 0.97
```

### 🔍 Interpretation

- The model achieves **97.13% accuracy**
- **Spam precision (0.99)** → very few false positives
- **Spam recall (0.79)** → most spam messages are correctly detected
- Balanced and realistic performance on an imbalanced dataset

---

## 🧪 Example Prediction

```python
predict_spam("Congratulations! You have won a free prize")
# Output: Spam
```

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone <repository-url>
cd spam-classifier
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Notebook

Open `spam_classifier.ipynb` in VS Code or Jupyter Notebook and run the cells sequentially.

---

## 📌 Future Improvements

- Deploy using **Streamlit**
- Try **Logistic Regression or SVM**
- Hyperparameter tuning
- Improve spam recall using threshold optimization
- Add real-time email/SMS input UI

---

## 💬 Interview Explanation (One-Liner)

> “I built a spam classification system using TF-IDF and Naive Bayes, achieving 97% accuracy while carefully handling class imbalance using stratified sampling.”

---

## 📜 License

This project is for **educational and learning purposes**.

---
