import streamlit as st
import pandas as pd

# Title
st.title("📩 Spam Message Classifier")
st.write ("This app predicts whether a message is spam or not spam  using Machine Learning.")
st.markdown("---")

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# ML model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
X = vectorizer.fit_transform(df['text'])
y = df['label']

model = LogisticRegression()
model.fit(X, y)

# User input
user_input = st.text_area("Enter your message:")

# Button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message")
    else:
        input_vector = vectorizer.transform([user_input])
        proba = model.predict_proba(input_vector)
        confidence = max(proba[0]) * 100

        if prediction[0] == 1:
            st.error("🚨 This is SPAM message")
        else:
            st.success("✅ This is NOT spam")

import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))