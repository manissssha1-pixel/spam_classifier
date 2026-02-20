import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")
# Keep only required columns
df = df[['v1', 'v2']]

print(df.head())
print(df.columns)

# Features and labels
X = df['v2']
y = df['v1']

# Convert labels
y = y.map({'ham': 0, 'spam': 1})

# Vectorize text
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vec, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model saved successfully!")