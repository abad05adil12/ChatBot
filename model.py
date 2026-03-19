import json
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

with open("data.json", "r") as f:
    data = json.load(f)

rows = []
for item in data:
    intent = item["intent"]
    for pattern in item["patterns"]:
        rows.append({"text": pattern, "intent": intent})

df = pd.DataFrame(rows)  
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

print(f"Dataset size: {len(df)} rows")
print(df.head())

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])

model = LogisticRegression(max_iter=200)
model.fit(X, df["intent"])

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model Trained Successfully!")