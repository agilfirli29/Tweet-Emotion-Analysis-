This project focuses on classifying **emotions from tweet texts** using a simple Machine Learning pipeline with **TF-IDF Vectorizer** and **Logistic Regression**.
The goal is to understand emotional patterns in social media text and demonstrate basic NLP modelling for portfolio and learning purposes.

---

## 📊 Dataset

* Dataset: **`tweet_emotions.csv`**
* Link : [https://www.kaggle.com/datasets/emirhanai/emotion-prediction-with-semi-supervised-learning]
* Contains around **40,000 tweets** labeled with various emotions such as:

* 😊 *Happiness*
* 😔 *Sadness*
* 😐 *Neutral*
* 😟 *Worry*
* 😡 *Anger*
* 💗 *Love*, and others.

---

## 🧹 Pre-processing

Steps applied before training:

* Removed duplicates and missing values
* Cleaned text (lowercase, removed URLs, punctuation, and special characters)
* Renamed column `content` → `text`
* Removed stopwords

---

## 🔍 Exploratory Data Analysis (EDA)

* Visualized emotion distribution using bar charts.
* Found that **neutral** and **worry** dominate the dataset.
* Minority classes (*anger*, *boredom*, *empty*) are underrepresented.

---

## ⚙️ Modelling

Model: **TF-IDF Vectorizer + Logistic Regression**
Configuration:

* `max_features = 5000`
* `ngram_range = (1, 2)`
* `max_iter = 1000`

The model was trained using 80% of the data and tested on 20% to evaluate generalization performance.

---

## 📈 Evaluation

**Metrics Used:**

* Accuracy
* Precision, Recall, F1-Score

**Result:**

* Accuracy ≈ **35%**
* Weighted F1-Score ≈ **0.31**

💡 *Interpretation:*
The model performs reasonably well on dominant classes like **neutral** and **worry**, but struggles on rare emotion classes. Future improvements could include data balancing, hyperparameter tuning, or fine-tuning transformer models (e.g., BERT).

---

## 🧰 Tools Used

*  Python
*  Pandas
*  Scikit-learn
*  Matplotlib
*  Seaborn
