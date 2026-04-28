# run_metrics.py
import joblib
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = joblib.load("fake_news_model.joblib")
df    = pd.read_csv("data.csv")

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.33, random_state=42
)

y_pred  = model.predict(X_test)
report  = classification_report(y_test, y_pred, output_dict=True)
cm      = confusion_matrix(y_test, y_pred).tolist()
acc     = accuracy_score(y_test, y_pred)

metrics = {
    "accuracy":         round(acc * 100, 1),
    "precision_fake":   round(report["fake"]["precision"]  * 100, 1),
    "recall_fake":      round(report["fake"]["recall"]     * 100, 1),
    "f1_fake":          round(report["fake"]["f1-score"]   * 100, 1),
    "precision_real":   round(report["real"]["precision"]  * 100, 1),
    "recall_real":      round(report["real"]["recall"]     * 100, 1),
    "f1_real":          round(report["real"]["f1-score"]   * 100, 1),
    "confusion_matrix": cm,
    "test_size":        len(y_test),
    "train_size":       len(y_train),
}

with open("model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Done! model_metrics.json saved.")