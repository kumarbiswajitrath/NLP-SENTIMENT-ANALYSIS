import joblib
import re


# Load trained model and TF-IDF vectorizer
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")


# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Take review from user
review = input("Enter a movie review: ")


# Clean the review
cleaned_review = clean_text(review)


# Convert review into TF-IDF features
review_tfidf = tfidf.transform([cleaned_review])


# Predict sentiment
prediction = model.predict(review_tfidf)


print("\nPredicted Sentiment:", prediction[0])