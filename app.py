import os
import datetime
import numpy as np
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import pipeline

MAX_PASSAGES = 3000
EMB_PATH = "models/app_passage_emb.npy"

st.set_page_config(page_title="Student Guide", page_icon="🎓")


@st.cache_resource(show_spinner="Loading models (first run takes a few minutes)...")
def load_resources():
    df = pd.read_csv("data/raw/sciq_train.csv").dropna(subset=["support"])
    passages = [p.strip() for p in dict.fromkeys(df["support"].tolist()) if p.strip()]
    passages = passages[:MAX_PASSAGES]

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    if os.path.exists(EMB_PATH) and np.load(EMB_PATH).shape[0] == len(passages):
        p_emb = np.load(EMB_PATH)
    else:
        p_emb = embedder.encode(passages, normalize_embeddings=True, show_progress_bar=True)
        np.save(EMB_PATH, p_emb)

    bm25 = BM25Okapi([p.lower().split() for p in passages])
    clf = pipeline("text-classification", model="models/distilbert_final",
                   tokenizer="models/distilbert_final")
    return passages, embedder, p_emb, bm25, clf


def norm(s):
    return (s - s.mean()) / (s.std() + 1e-9)


def search(question, method, k):
    bm = bm25.get_scores(question.lower().split())
    em = (embedder.encode([question], normalize_embeddings=True) @ p_emb.T)[0]
    if method == "BM25 (keyword)":
        scores = bm
    elif method == "Embeddings (semantic)":
        scores = em
    else:
        scores = 0.5 * norm(bm) + 0.5 * norm(em)
    top = np.argsort(-scores)[:k]
    return [passages[i] for i in top]


def save_row(path, row):
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


passages, embedder, p_emb, bm25, clf = load_resources()

st.title("🎓 Student Guide")
st.caption("Intelligent question classification and answer recommendation")

tab_ask, tab_sus = st.tabs(["Ask a question", "Usability survey"])

# ---------------- Tab 1: ask ----------------
with tab_ask:
    method = st.sidebar.selectbox(
        "Retrieval method",
        ["BM25 (keyword)", "Embeddings (semantic)", "Hybrid"])
    k = st.sidebar.slider("Number of results", 1, 5, 3)

    question = st.text_input("Type your question",
                             placeholder="e.g. What organelle produces energy in a cell?")

    if st.button("Search") and question.strip():
        label = clf(question)[0]
        st.session_state["result"] = {
            "question": question,
            "category": label["label"],
            "confidence": label["score"],
            "method": method,
            "passages": search(question, method, k),
        }

    res = st.session_state.get("result")
    if res:
        st.subheader("Predicted subject")
        st.success(f"{res['category'].replace('_', ' ').title()} "
                   f"(confidence {res['confidence']:.0%})")

        st.subheader("Recommended resources")
        for i, p in enumerate(res["passages"], 1):
            with st.expander(f"Result {i}: {p[:90]}...", expanded=(i == 1)):
                st.write(p)

        with st.form("feedback"):
            helpful = st.radio("Were these results helpful?", ["Yes", "Somewhat", "No"],
                               horizontal=True)
            if st.form_submit_button("Send feedback"):
                save_row("data/feedback.csv", {
                    "time": datetime.datetime.now().isoformat(timespec="seconds"),
                    "question": res["question"], "category": res["category"],
                    "method": res["method"], "helpful": helpful})
                st.success("Thanks for your feedback!")

# ---------------- Tab 2: SUS ----------------
SUS = [
    "I think that I would like to use this system frequently.",
    "I found the system unnecessarily complex.",
    "I thought the system was easy to use.",
    "I think that I would need the support of a technical person to use this system.",
    "I found the various functions in this system were well integrated.",
    "I thought there was too much inconsistency in this system.",
    "I would imagine that most people would learn to use this system very quickly.",
    "I found the system very cumbersome to use.",
    "I felt very confident using the system.",
    "I needed to learn a lot of things before I could get going with this system.",
]

with tab_sus:
    st.write("Please try the system first, then rate each statement "
             "(1 = strongly disagree, 5 = strongly agree).")
    with st.form("sus"):
        answers = [st.slider(f"{i}. {q}", 1, 5, 3) for i, q in enumerate(SUS, 1)]
        if st.form_submit_button("Submit survey"):
            # Odd items: score - 1. Even items: 5 - score. Total x 2.5 -> 0-100
            total = sum((a - 1) if i % 2 == 1 else (5 - a)
                        for i, a in enumerate(answers, 1))
            sus_score = total * 2.5
            row = {"time": datetime.datetime.now().isoformat(timespec="seconds"),
                   "sus_score": sus_score}
            row.update({f"q{i}": a for i, a in enumerate(answers, 1)})
            save_row("data/sus_responses.csv", row)
            st.success(f"Thank you! Your SUS score: {sus_score:.1f} / 100")