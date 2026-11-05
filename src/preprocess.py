import re
import pandas as pd
import spacy
from sklearn.model_selection import train_test_split

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

df = pd.read_csv("data/raw/mmlu_selected.csv")
df = df[["question", "subject"]].dropna().drop_duplicates()

def light_clean(text):
    # For the Transformer: keep the wording, just tidy whitespace
    return re.sub(r"\s+", " ", str(text)).strip()

def heavy_clean(text):
    # For TF-IDF: lowercase, remove noise, remove stop words, lemmatize
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", str(text).lower())
    doc = nlp(text)
    tokens = [t.lemma_ for t in doc if not t.is_stop and not t.is_space and len(t) > 1]
    return " ".join(tokens)

df["text_light"] = df["question"].apply(light_clean)
df["text_heavy"] = df["question"].apply(heavy_clean)

# Numeric labels for the models
labels = sorted(df["subject"].unique())
label2id = {name: i for i, name in enumerate(labels)}
df["label"] = df["subject"].map(label2id)

# 70% train, 15% validation, 15% test, keeping class proportions equal
train, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
val, test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=42)

train.to_csv("data/processed/train.csv", index=False)
val.to_csv("data/processed/val.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

print("Classes:", label2id)
print("Sizes -> train:", len(train), "val:", len(val), "test:", len(test))
print(train[["text_light", "text_heavy", "subject"]].head())