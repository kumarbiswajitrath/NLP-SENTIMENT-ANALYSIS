from transformers import pipeline


# ============================================================
# 1. LOAD PRETRAINED SENTIMENT MODEL
# ============================================================

print("Loading pretrained sentiment model...")

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

print("Model loaded successfully!")


# ============================================================
# 2. TAKE USER INPUT
# ============================================================

review = input("\nEnter a movie review: ")


# ============================================================
# 3. PREDICT SENTIMENT
# ============================================================

result = sentiment_pipeline(review)


# ============================================================
# 4. DISPLAY RESULT
# ============================================================

label = result[0]["label"]
score = result[0]["score"]

print("\n========================================")
print("SENTIMENT ANALYSIS RESULT")
print("========================================")

print("Review:", review)

print("Predicted Sentiment:", label)

print("Confidence:", round(score * 100, 2), "%")