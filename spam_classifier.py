import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("spam.csv", encoding="latin-1")
print(df.head())
df = df[['v1', 'v2']]
df.columns = ['label', 'text']
print(df.head())
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
X = vectorizer.fit_transform(df['text'])
y = df['label']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(classification_report(y_test, y_pred))
sample = ["Congratulations! You have won a free iPhone. Click here now!"]
sample_vector = vectorizer.transform(sample)
prediction = model.predict(sample_vector)

print("Prediction:", prediction)
sample2 = ["Hey, are we meeting tomorrow at 5 pm?"]
sample2_vector = vectorizer.transform(sample2)
prediction2 = model.predict(sample2_vector)
print("Prediction 2:", prediction2)

sample3 = ["URGENT! You have won $1000 cash prize. Click link now!"]
sample3_vector = vectorizer.transform(sample)
prediction = model.predict(sample_vector)