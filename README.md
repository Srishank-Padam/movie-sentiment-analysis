# 🎬 Movie Review Sentiment Analysis (Streamlit App)

This is a **Movie Review Sentiment Analysis Web App** built using **Python, NLP, Scikit-learn, and Streamlit**.  
The app predicts whether a given movie review is **Positive** or **Negative** using a trained **TF-IDF + Logistic Regression** model.  

---

## 📌 Project Overview
- **Objective**: Classify movie reviews into positive or negative sentiments.  
- **Dataset**: User-provided CSV file (Kaggle dataset with `review` and `sentiment` columns).  
- **Tech Stack**:  
  - Python  
  - Pandas, Scikit-learn, NLTK  
  - Streamlit  

---

## 📂 Project Structure
```
Movie-Sentiment-Analysis/
│── train.py # Script to train the model
│── app.py # Streamlit web app for predictions
│── requirements.txt # Dependencies
│── README.md # Project documentation
│── models/ # Stores trained model + vectorizer
│── data/ # <-- (You need to add this manually before running)
```

---

## ⚙️ Installation
### 1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/Movie-Sentiment-Analysis.git
   cd Movie-Sentiment-Analysis
   ```
   
