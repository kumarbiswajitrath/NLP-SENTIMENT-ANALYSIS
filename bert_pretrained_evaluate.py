import pandas as pd
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from transformers import pipeline


# ============================================================
# 1. LOAD IMDB DATASET
# ============================================================

print("\nLoading IMDB dataset...")

path = kagglehub.dataset_download(
    "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
)

file_path = path + "/IMDB Dataset.csv"

data = pd.read_csv(file_path)

print("Dataset loaded!")
print("Dataset shape:", data.shape)


# ============================================================
# 2. CONVERT SENTIMENT TO NUMERICAL LABEL
# ============================================================

data["label"] = data["sentiment"].map({
    "negative": 0,
    "positive": 1
})


# ============================================================
# 3. CREATE TEST SET
# ============================================================

_, test_data = train_test_split(
    data,
    test_size=1000,
    random_state=42,
    stratify=data["label"]
)

print("\nTest samples:", len(test_data))

print("\nTest label counts:")
print(test_data["label"].value_counts())


# ============================================================
# 4. LOAD PRETRAINED DISTILBERT
# ============================================================

print("\n========================================")
print("LOADING PRETRAINED DISTILBERT")
print("========================================")

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

print("Pretrained model loaded!")


# ============================================================
# 5. MAKE PREDICTIONS
# ============================================================

print("\n========================================")
print("MAKING PREDICTIONS")
print("========================================")

reviews = test_data["review"].tolist()

results = []

batch_size = 16

for i in range(0, len(reviews), batch_size):

    batch = reviews[i:i + batch_size]

    predictions = sentiment_pipeline(
        batch,
        truncation=True,
        max_length=256
    )

    results.extend(predictions)

    print(
        f"Processed {min(i + batch_size, len(reviews))}"
        f"/{len(reviews)}"
    )


# ============================================================
# 6. CONVERT PREDICTIONS TO 0/1
# ============================================================

predicted_labels = []

for result in results:

    if result["label"] == "NEGATIVE":
        predicted_labels.append(0)
    else:
        predicted_labels.append(1)


true_labels = test_data["label"].tolist()


# ============================================================
# 7. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

f1 = f1_score(
    true_labels,
    predicted_labels
)


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("PRETRAINED DISTILBERT RESULTS")
print("========================================")

print("Accuracy:", accuracy)

print("Accuracy %:", accuracy * 100)

print("F1 Score:", f1)


# ============================================================
# 9. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=[
            "negative",
            "positive"
        ],
        zero_division=0
    )
)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

print(cm)


# ============================================================
# 11. FINISHED
# ============================================================

print("\n========================================")
print("EVALUATION COMPLETED")
print("========================================")