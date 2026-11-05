import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)

train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")
test = pd.read_csv("data/processed/test.csv")

names = [train[train["label"] == i]["subject"].iloc[0] for i in sorted(train["label"].unique())]
id2label = {i: n for i, n in enumerate(names)}
label2id = {n: i for i, n in id2label.items()}

checkpoint = "distilbert-base-uncased"
tok = AutoTokenizer.from_pretrained(checkpoint)

def to_ds(df):
    ds = Dataset.from_pandas(df[["text_light", "label"]].rename(columns={"text_light": "text"}))
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=128), batched=True)

train_ds, val_ds, test_ds = to_ds(train), to_ds(val), to_ds(test)

model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=len(names), id2label=id2label, label2id=label2id)

def metrics(p):
    return {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))}

args = TrainingArguments(
    output_dir="models/distilbert_out",
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=3e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",
)

trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                  processing_class=tok, compute_metrics=metrics)
trainer.train()

out = trainer.predict(test_ds)
pred = np.argmax(out.predictions, axis=1)
print("Test accuracy:", round(accuracy_score(test["label"], pred), 3))
print(classification_report(test["label"], pred, target_names=names))
print(confusion_matrix(test["label"], pred))

trainer.save_model("models/distilbert_final")
tok.save_pretrained("models/distilbert_final")