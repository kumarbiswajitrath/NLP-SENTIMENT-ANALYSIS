
import streamlit as st
import joblib
import re
from transformers import pipeline


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Movie Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)


# =========================================================
# LOAD LOGISTIC REGRESSION MODEL
# =========================================================

@st.cache_resource
def load_logistic_model():

    model = joblib.load("sentiment_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")

    return model, tfidf


# =========================================================
# LOAD PRETRAINED DISTILBERT
# =========================================================

@st.cache_resource
def load_distilbert():

    sentiment_model = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    return sentiment_model


# =========================================================
# CLEAN TEXT FOR LOGISTIC REGRESSION
# =========================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"<br\s*/?>",
        " ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# =========================================================
# LOAD MODELS
# =========================================================

model, tfidf = load_logistic_model()

distilbert = load_distilbert()


# =========================================================
# TITLE
# =========================================================

st.title("🎬 Movie Sentiment Analysis")

st.markdown(
    "### Compare two NLP sentiment analysis models"
)

st.write(
    "This application analyzes a movie review using "
    "TF-IDF + Logistic Regression and a pretrained DistilBERT model."
)


# =========================================================
# REVIEW INPUT
# =========================================================

st.subheader("Enter a Movie Review")

review = st.text_area(
    "Movie review",
    height=180,
    placeholder=(
        "Example: The story was engaging, "
        "the acting was excellent, and I really enjoyed the movie."
    ),
    label_visibility="collapsed"
)


# =========================================================
# ANALYZE BUTTON
# =========================================================

if st.button(
    "🔍 Analyze Sentiment",
    use_container_width=True
):

    if review.strip() == "":

        st.warning(
            "Please enter a movie review before analyzing."
        )

    else:

        # =================================================
        # LOGISTIC REGRESSION
        # =================================================

        cleaned_review = clean_text(review)

        review_tfidf = tfidf.transform(
            [cleaned_review]
        )

        lr_prediction = model.predict(
            review_tfidf
        )[0]

        probabilities = model.predict_proba(
            review_tfidf
        )[0]

        classes = list(model.classes_)

        negative_index = classes.index("negative")
        positive_index = classes.index("positive")

        negative_probability = probabilities[
            negative_index
        ]

        positive_probability = probabilities[
            positive_index
        ]

        if lr_prediction == "positive":

            lr_confidence = positive_probability

        else:

            lr_confidence = negative_probability


        # =================================================
        # DISTILBERT
        # =================================================

        bert_result = distilbert(
            review,
            truncation=True,
            max_length=256
        )[0]

        bert_label = bert_result["label"]
        bert_confidence = bert_result["score"]


        # =================================================
        # RESULTS
        # =================================================

        st.divider()

        st.header("📊 Analysis Results")


        # =================================================
        # TWO COLUMNS
        # =================================================

        col1, col2 = st.columns(2)


        # =================================================
        # LOGISTIC REGRESSION RESULT
        # =================================================

        with col1:

            st.subheader(
                "TF-IDF + Logistic Regression"
            )

            if lr_prediction == "positive":

                st.success(
                    "😊 POSITIVE"
                )

            else:

                st.error(
                    "😞 NEGATIVE"
                )

            st.metric(
                "Confidence",
                f"{lr_confidence * 100:.2f}%"
            )

            st.progress(
                float(lr_confidence)
            )

            st.write(
                f"Positive probability: "
                f"{positive_probability * 100:.2f}%"
            )

            st.write(
                f"Negative probability: "
                f"{negative_probability * 100:.2f}%"
            )


        # =================================================
        # DISTILBERT RESULT
        # =================================================

        with col2:

            st.subheader(
                "Pretrained DistilBERT"
            )

            if bert_label == "POSITIVE":

                st.success(
                    "😊 POSITIVE"
                )

            else:

                st.error(
                    "😞 NEGATIVE"
                )

            st.metric(
                "Confidence",
                f"{bert_confidence * 100:.2f}%"
            )

            st.progress(
                float(bert_confidence)
            )


        # =================================================
        # REVIEW
        # =================================================

        st.divider()

        st.subheader("📝 Your Review")

        st.info(review)


# =========================================================
# PROJECT INFORMATION
# =========================================================

st.divider()

st.subheader("About the Project")

st.write(
    """
This project performs movie sentiment analysis using two
different NLP approaches:

• TF-IDF + Logistic Regression  
• Pretrained DistilBERT

The Logistic Regression model was trained on the IMDB
movie review dataset, while DistilBERT uses a pretrained
sentiment classification model.
"""
)

