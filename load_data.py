import pandas as pd
import kagglehub

path = kagglehub.dataset_download(
    "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
)

file_path = path + "/IMDB Dataset.csv"

data = pd.read_csv(file_path)

print(data.head())
print("\nShape of dataset:")
print(data.shape)

print("\nColumn names:")
print(data.columns)
print("\nSentiment counts:")
print(data["sentiment"].value_counts())