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
   git clone https://github.com/Srishank-Padam/movie-sentiment-analysis
   cd movie-sentiment-analysis
   ```

### 2. Create a virtual environment & install requirements:
```bash
python -m venv .venv
source .venv/bin/activate    # (Linux/Mac)
.venv\Scripts\activate       # (Windows)

pip install -r requirements.txt
```

## 📂 Adding the Dataset
```
Since the dataset is not included in this repo, you need to:

Create a data/ folder inside the project directory.

Download the Kaggle dataset (IMDB Dataset or similar with review & sentiment columns).

Place the CSV file inside the data/ folder. Example:
```
```bash
Movie-Sentiment-Analysis/
└── data/
    └── movie_reviews.csv
```

## 🚀 Running the Project
### 1. Train the Model
```
Run the training script with your dataset:
```
```bash
python train.py --csv data/movie_reviews.csv --out_dir models/
```

### 2. Run the Streamlit App
```
Start the web app:
```
```bash
streamlit run app.py
```
```
Open the URL shown in the terminal (usually http://localhost:8501/) in your browser.
```

## 🖼️ Example Usage
```
Input:
“This movie was absolutely fantastic! The acting was great and the story was engaging.”

Output:
Predicted Sentiment → Positive ✅
```
## 🔮 Future Improvements
```
Add more ML/DL models (Naive Bayes, LSTMs, Transformers).

Support multi-class classification (positive, negative, neutral).

Deploy on Streamlit Cloud / Heroku / AWS.
```

## 👤 Author
```
Developed by [Padam Srishank]
```

   
