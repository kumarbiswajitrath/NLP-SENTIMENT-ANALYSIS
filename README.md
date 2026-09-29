# 🎬 NLP Sentiment Analysis

A machine learning and NLP project for classifying movie reviews as **positive** or **negative**.

This project implements and compares two approaches:

1. **TF-IDF + Logistic Regression**
2. **Pretrained DistilBERT**

A **Streamlit web application** allows users to enter a movie review and see predictions from both models.

---

## 📌 Project Overview

Sentiment analysis is an NLP task used to determine the emotional tone of text.

In this project, movie reviews from the **IMDB 50K Movie Reviews dataset** are processed and classified into:

* **Positive**
* **Negative**

The project covers the complete workflow:

```text
Movie Reviews
      ↓
Text Preprocessing
      ↓
Train/Test Split
      ↓
TF-IDF Feature Extraction
      ↓
Logistic Regression
      ↓
Prediction & Evaluation
```

A pretrained Transformer model is also used:

```text
Movie Review
      ↓
Pretrained DistilBERT
      ↓
Sentiment Prediction
```

---

## 📂 Dataset

**Dataset:** IMDB Dataset of 50K Movie Reviews

* Total reviews: **50,000**
* Positive reviews: **25,000**
* Negative reviews: **25,000**
* Features:

  * `review`
  * `sentiment`

The dataset is downloaded using KaggleHub.

The dataset itself is not included in this repository.

---

## 🧹 Text Preprocessing

The following preprocessing steps are applied to the reviews:

* Convert text to lowercase
* Remove HTML `<br>` tags
* Remove non-alphabetic characters
* Remove extra spaces

Example:

```text
Original:
"This movie was AMAZING!!! <br /> I loved it."

After preprocessing:
"this movie was amazing i loved it"
```

---

# 🤖 Model 1: TF-IDF + Logistic Regression

### TF-IDF

TF-IDF converts text into numerical features based on the importance of words in the documents.

The project uses:

```python
TfidfVectorizer(
    max_features=20000,
    stop_words="english"
)
```

The training data contains:

```text
40,000 reviews × 20,000 TF-IDF features
```

The test data contains:

```text
10,000 reviews × 20,000 TF-IDF features
```

### Logistic Regression

The TF-IDF features are given to a Logistic Regression classifier.

```python
LogisticRegression(max_iter=1000)
```

### Measured Results

The model achieved:

| Metric   |     Result |
| -------- | ---------: |
| Accuracy | **89.72%** |
| F1 Score | **≈ 0.90** |

Classification results:

```text
              precision    recall  f1-score   support

negative        0.91      0.89      0.90      5000
positive        0.89      0.91      0.90      5000

accuracy                            0.90     10000
```

### Confusion Matrix

The confusion matrix obtained during evaluation was:

```text
[[4436  564]
 [ 464 4536]]
```

The corresponding visualization is included in:

```text
confusion_matrix.png
```

---

# 🧠 Model 2: Pretrained DistilBERT

The project also uses the pretrained:

```text
distilbert-base-uncased-finetuned-sst-2-english
```

DistilBERT is a Transformer-based language model that has already been trained for sentiment classification.

Unlike the Logistic Regression approach, the review is directly passed through the pretrained Transformer model.

### Evaluation

The pretrained DistilBERT model was evaluated on **1,000 balanced IMDB test reviews**.

Results:

| Metric   |     Result |
| -------- | ---------: |
| Accuracy | **87.80%** |
| F1 Score | **0.8753** |

Classification results:

```text
              precision    recall  f1-score   support

negative        0.86      0.90      0.88       500
positive        0.90      0.86      0.88       500

accuracy                            0.88      1000
```

Confusion matrix:

```text
[[450  50]
 [ 72 428]]
```

---

# 📊 Model Results

The measured evaluation results are summarized below:

| Model                        | Evaluation Set |   Accuracy |   F1 Score |
| ---------------------------- | -------------: | ---------: | ---------: |
| TF-IDF + Logistic Regression | 10,000 reviews | **89.72%** | **≈ 0.90** |
| Pretrained DistilBERT        |  1,000 reviews | **87.80%** | **0.8753** |

The evaluation sets are different sizes, so the results should be interpreted with that difference in mind.

---

# 🌐 Streamlit Web Application

The project includes an interactive Streamlit application.

The application allows users to:

* Enter a movie review
* Get a prediction from TF-IDF + Logistic Regression
* Get a prediction from pretrained DistilBERT
* View the confidence reported for each prediction
* Compare the two model outputs

Run the application using:

```powershell
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

---

## 🧪 Example Predictions

### Clearly Positive Review

```text
The movie was amazing from beginning to end.
The story was engaging, the acting was excellent,
and the characters felt realistic. Definitely worth watching.
```

Both models predicted:

```text
POSITIVE
```

with high confidence.

### Clearly Negative Review

```text
This movie was terrible. The story was boring,
the acting was weak, and the characters were poorly written.
I would not recommend it.
```

Both models predicted:

```text
NEGATIVE
```

with high confidence.

### Mixed Review

```text
The movie had some excellent performances and a few
really enjoyable scenes, but the story was slow and predictable.
The first half was interesting, while the second half felt too long.
Overall, it was an average movie.
```

The two models produced different predictions:

```text
TF-IDF + Logistic Regression
→ POSITIVE — 65.58%

Pretrained DistilBERT
→ NEGATIVE — 52.91%
```

This demonstrates that ambiguous reviews can produce less confident and different predictions.

---

# 📁 Project Structure

```text
NLP-SENTIMENT-ANALYSIS/
│
├── app.py
├── train.py
├── predict.py
├── evaluate.py
├── preprocess.py
├── load_data.py
│
├── bert_predict.py
├── bert_pretrained_evaluate.py
│
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── confusion_matrix.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TF-IDF
* Logistic Regression
* PyTorch
* Hugging Face Transformers
* DistilBERT
* Streamlit
* Matplotlib
* Seaborn
* KaggleHub
* Git & GitHub

---

# ▶️ How to Run

### 1. Clone the repository

```powershell
git clone https://github.com/kumarbiswajitrath/NLP-SENTIMENT-ANALYSIS.git
```

### 2. Enter the project directory

```powershell
cd NLP-SENTIMENT-ANALYSIS
```

### 3. Create a virtual environment

```powershell
python -m venv venv
```

### 4. Activate the environment

```powershell
venv\Scripts\activate
```

### 5. Install dependencies

```powershell
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```powershell
streamlit run app.py
```

---

# 📌 Key Concepts Demonstrated

This project demonstrates practical understanding of:

* Natural Language Processing
* Text preprocessing
* Train/test splitting
* TF-IDF feature extraction
* Sparse feature matrices
* Logistic Regression classification
* Transformer-based NLP
* DistilBERT
* Model evaluation
* Accuracy and F1 score
* Classification reports
* Confusion matrices
* Streamlit deployment
* Git and GitHub project management

---

# 🚀 Future Improvements

Possible extensions include:

* Hyperparameter tuning
* Larger-scale Transformer fine-tuning
* Additional NLP preprocessing experiments
* Comparison with other ML classifiers
* More extensive evaluation datasets
* Deployment of the Streamlit application
* Analysis of model errors and ambiguous reviews

---

## 👨‍💻 Author

**Biswajit Rath**

Electronics & Communication Engineering
National Institute of Technology Rourkela

GitHub:

https://github.com/kumarbiswajitrath
