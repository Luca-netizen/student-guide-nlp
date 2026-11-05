import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ---------- Data: SciQ questions that have a support passage ----------
df = pd.read_csv("data/raw/sciq_train.csv")
df = df.dropna(subset=["question", "support", "correct_answer"])
df = df[df["support"].str.strip() != ""]
df = df.sample(n=min(2000, len(df)), random_state=42).reset_index(drop=True)

questions = df["question"].tolist()
passages = list(dict.fromkeys(df["support"].tolist()))   # unique passages
pid = {p: i for i, p in enumerate(passages)}
true_idx = np.array([pid[p] for p in df["support"]])
print("Questions:", len(questions), "| Candidate passages:", len(passages))

# ---------- Metrics ----------
def evaluate(scores, name):
    # rank of the correct passage = 1 + number of passages scored higher
    true_scores = scores[np.arange(len(scores)), true_idx][:, None]
    ranks = (scores > true_scores).sum(axis=1) + 1
    print(f"{name:12s} Top-1: {np.mean(ranks <= 1):.3f}  Top-3: {np.mean(ranks <= 3):.3f}  "
          f"Top-5: {np.mean(ranks <= 5):.3f}  MRR: {np.mean(1 / ranks):.3f}")

# ---------- Method 1: BM25 keyword baseline ----------
tokenize = lambda t: t.lower().split()
bm25 = BM25Okapi([tokenize(p) for p in passages])
bm25_scores = np.array([bm25.get_scores(tokenize(q)) for q in questions])
evaluate(bm25_scores, "BM25")

# ---------- Method 2: sentence embeddings ----------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
p_emb = embedder.encode(passages, normalize_embeddings=True, show_progress_bar=True)
q_emb = embedder.encode(questions, normalize_embeddings=True, show_progress_bar=True)
emb_scores = q_emb @ p_emb.T
evaluate(emb_scores, "Embeddings")

# ---------- Demo: classify, then recommend ----------
clf = pipeline("text-classification", model="models/distilbert_final",
               tokenizer="models/distilbert_final")

def recommend(question, k=3):
    label = clf(question)[0]
    print(f"\nQuestion: {question}")
    print(f"Predicted category: {label['label']} ({label['score']:.2f})")
    q = embedder.encode([question], normalize_embeddings=True)
    top = np.argsort(-(q @ p_emb.T)[0])[:k]
    for rank, i in enumerate(top, 1):
        print(f"  {rank}. {passages[i][:200]}...")

recommend("What organelle produces energy in a cell?")
recommend("What type of bond forms when atoms share electrons?")
recommend("What force pulls objects toward the center of the Earth?")