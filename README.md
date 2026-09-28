# NLP Sentiment Analysis

An NLP-based sentiment analysis project that classifies IMDB movie reviews as **positive** or **negative** using **TF-IDF vectorization** and **Logistic Regression**.

## Project Overview

This project implements an end-to-end Natural Language Processing (NLP) pipeline for movie review sentiment classification.

The workflow includes:

* Loading the IMDB movie review dataset
* Text preprocessing and cleaning
* Train-test splitting
* TF-IDF feature extraction
* Logistic Regression model training
* Model evaluation
* Confusion matrix visualization
* Prediction on custom movie reviews

## Dataset

The project uses the **IMDB Dataset of 50K Movie Reviews**.

* Total reviews: **50,000**
* Positive reviews: **25,000**
* Negative reviews: **25,000**
* Training samples: **40,000**
* Testing samples: **10,000**

The dataset is not included in this repository.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* KaggleHub

## NLP Preprocessing

The reviews are cleaned using the following steps:

1. Convert text to lowercase
2. Remove HTML `<br>` tags
3. Remove non-alphabetic characters
4. Remove extra whitespace

## TF-IDF Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert the cleaned text into numerical features.

Configuration:

* Maximum features: **20,000**
* Stop words: **English**

The resulting matrices are:

```text
Training: (40000, 20000)
Testing:  (10000, 20000)
```

This means that each review is represented using up to **20,000 TF-IDF features**.

## Machine Learning Model

A **Logistic Regression** classifier is used for sentiment classification.

The model is trained using the TF-IDF representation of the training reviews.

## Results

The model was evaluated on **10,000 test reviews**.

### Accuracy

**89.72%**

### Classification Report

| Class    | Precision | Recall | F1-score |
| -------- | --------: | -----: | -------: |
| Negative |      0.91 |   0.89 |     0.90 |
| Positive |      0.89 |   0.91 |     0.90 |

### Confusion Matrix

```text
                 Predicted
               Negative  Positive
Actual Negative   4436      564
       Positive    464     4536
```

The confusion matrix visualization is available in:

```text
confusion_matrix.png
```

## Project Structure

```text
NLP-SENTIMENT-ANALYSIS/
│
├── load_data.py
├── preprocess.py
├── train.py
├── evaluate.py
├── predict.py
├── requirements.txt
├── .gitignore
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
└── confusion_matrix.png
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/kumarbiswajitrath/NLP-SENTIMENT-ANALYSIS.git
```

### 2. Open the project directory

```bash
cd NLP-SENTIMENT-ANALYSIS
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Train the model

```bash
python train.py
```

This creates:

```text
sentiment_model.pkl
tfidf_vectorizer.pkl
```

### 7. Evaluate the model

```bash
python evaluate.py
```

This displays the accuracy, classification report, and confusion matrix.

### 8. Predict sentiment for a custom review

```bash
python predict.py
```

Enter a movie review when prompted.

Example:

```text
Enter a movie review: This movie was fantastic and I really enjoyed it.

Predicted Sentiment: positive
```

## Model Performance

The Logistic Regression model achieved an accuracy of:

**89.72%**

on the 10,000-review test set.

## Future Improvements

Possible extensions to this project include:

* Comparing Logistic Regression with Naive Bayes
* Using word n-grams and character n-grams
* Hyperparameter tuning
* Implementing a BERT-based sentiment classifier
* Developing a Streamlit web interface
* Comparing traditional NLP methods with Transformer-based models
