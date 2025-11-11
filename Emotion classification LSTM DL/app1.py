# emotion_app.py
import streamlit as st
import pickle
import re
from nltk.stem.porter import PorterStemmer
import nltk

st.title("Test Streamlit App")
st.write("If you see this, Streamlit is working!")

# Download stopwords once
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))

# -----------------------------
# 1. Load model, vectorizer, encoder
# -----------------------------
with open("logistic_regression.pkl", "rb") as f:
    lg = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    lb = pickle.load(f)

# -----------------------------
# 2. Text cleaning function
# -----------------------------
def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)

# -----------------------------
# 3. Prediction function
# -----------------------------
def predict_emotion(text):
    cleaned_text = clean_text(text)
    vector = tfidf_vectorizer.transform([cleaned_text])
    pred_label = lg.predict(vector)[0]
    emotion = lb.inverse_transform([pred_label])[0]
    return emotion

# -----------------------------
# 4. Streamlit UI
# -----------------------------
st.title("📝 Emotion Detection App")
st.write("Enter a sentence to predict its emotion:")

user_input = st.text_area("Type your text here:")

if st.button("Predict Emotion"):
    if user_input.strip() != "":
        emotion = predict_emotion(user_input)
        st.success(f"Predicted Emotion: {emotion}")
    else:
        st.warning("Please enter some text!")
