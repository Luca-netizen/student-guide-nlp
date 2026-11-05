import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")
test = pd.read_csv("data/processed/test.csv")

tfidf = joblib.load("models/tfidf.joblib")
X_train = tfidf.transform(train["text_heavy"].fillna(""))
X_val = tfidf.transform(val["text_heavy"].fillna(""))
X_test = tfidf.transform(test["text_heavy"].fillna(""))

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "LinearSVC": LinearSVC(),
}

best_name, best_acc = None, 0
for name, clf in models.items():
    clf.fit(X_train, train["label"])
    acc = accuracy_score(val["label"], clf.predict(X_val))
    print(f"{name} validation accuracy: {acc:.3f}")
    if acc > best_acc:
        best_name, best_acc = name, acc

print("Best baseline:", best_name)
best = models[best_name]
joblib.dump(best, "models/baseline.joblib")

# Final evaluation on the test set
pred = best.predict(X_test)
names = [train[train["label"] == i]["subject"].iloc[0] for i in sorted(train["label"].unique())]
print("Test accuracy:", round(accuracy_score(test["label"], pred), 3))
print(classification_report(test["label"], pred, target_names=names))

cm = confusion_matrix(test["label"], pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=names, yticklabels=names, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Baseline confusion matrix (test set)")
plt.tight_layout()
plt.savefig("models/baseline_confusion.png")