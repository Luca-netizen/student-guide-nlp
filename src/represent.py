import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")
test = pd.read_csv("data/processed/test.csv")

# 1) TF-IDF for the baseline (fit on training data only, to avoid leakage)
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)
X_train = tfidf.fit_transform(train["text_heavy"].fillna(""))
print("TF-IDF train shape:", X_train.shape)
joblib.dump(tfidf, "models/tfidf.joblib")

# 2) Sentence embeddings for retrieval (use the lightly cleaned text)
model = SentenceTransformer("all-MiniLM-L6-v2")
for name, d in [("train", train), ("val", val), ("test", test)]:
    emb = model.encode(
        d["text_light"].tolist(),
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    np.save(f"data/processed/emb_{name}.npy", emb)
    print(name, "embeddings:", emb.shape)