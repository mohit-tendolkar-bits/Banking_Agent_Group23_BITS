"""Classical reference: TF-IDF + logistic regression, trained on the full train split
and scored on the 100 development rows. This is the bar the LLM must beat.

Run from the project root: uv run -m evaluation.reference_classifier"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

train = pd.read_csv("data/train.csv")
data = pd.read_csv("evaluation/eval_subset.csv")

classifier = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=1000))
classifier.fit(train["text"], train["category"])

data["predicted"] = classifier.predict(data["text"])
data["correct"] = data["predicted"] == data["category"]
data.to_csv("evaluation/results/results_tfidf_logreg.csv", index=False)

print(f"trained on {len(train)} train rows, scored on {len(data)} development rows")
print(f"accuracy: {data['correct'].mean():.2f}")
