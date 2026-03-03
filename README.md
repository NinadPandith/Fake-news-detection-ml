# 📰 AI Fake News Detection App

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

A robust Machine Learning project to detect whether a given news article is **Real** or **Fake**. The application uses Natural Language Processing (NLP) techniques along with a Logistic Regression model to classify news text and explain its predictions.

## 🌟 Project Overview
Fake news spreads rapidly on the internet, causing misinformation. This project aims to combat that by automatically identifying deceptive articles. The model evaluates the text, provides a prediction with dynamic confidence scores, and extracts the top words influencing the **Fake** classification for full explainability.

---

## 🛠️ Tech Stack
- **Language**: Python
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn
- **Text Processing**: TF-IDF Vectorizer
- **Frontend / UI**: Streamlit

---

## 📊 Dataset & Model Performance
- **Dataset Size**: ~44,000 samples combined from `Fake.csv` and `True.csv`.
- **Algorithm**: Logistic Regression
- **Data Split**: 80% Training / 20% Testing
- **Accuracy**: **98.5%**

---

## 🚀 How to Run Locally

### Prerequisites
Make sure you have Python installed. Then, install the required dependencies:
```bash
pip install pandas numpy scikit-learn streamlit joblib
```

### 1. Train the Model (Optional)
If you need to retrain the model (make sure you have `Fake.csv` and `True.csv` in the root folder):
```bash
python train_model.py
```
This will generate `model.pkl` and `vectorizer.pkl`.

### 2. Run the Web Application
Launch the Streamlit frontend to interact with the model:
```bash
python -m streamlit run app.py
```
The application will open in your default web browser at `http://localhost:8501`.

---

## 📸 Screenshots
*(Add screenshots of your application here)*

- **Prediction Result**
  ![Prediction Screenshot](#)

- **Feature Explainability**
  ![Explainability Screenshot](#)

---

## 🔮 Future Improvements
- Implement deep learning architectures (LSTMs, BERT) for improved context understanding.
- Add multi-language support.
- Scrape live articles via URL for real-time fake news detection.
- Deploy the application to a cloud provider (AWS, Heroku, or Streamlit Community Cloud).

---
*Built by Ninad using Machine Learning*
