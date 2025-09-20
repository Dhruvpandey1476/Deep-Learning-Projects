# sentiment_lstm_app.py

import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------
# 1. Load model and tokenizer
# -------------------------


model = load_model("project/sentiment_model.keras")

with open("project/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# same maxlen you used during training
MAXLEN = 200  

# -------------------------
# 2. Streamlit UI
# -------------------------
st.title("📊 Sentiment Analysis with LSTM")
st.write("Enter a sentence and see whether it's **Positive** or **Negative**")
# ...existing code...
MAXLEN = 200  # Match training

def preprocess_text(text):
    return text.lower().strip()  # Add more cleaning if needed

user_input = st.text_area("Enter text here:")

if st.button("Analyze Sentiment"):
    if user_input.strip() != "":
        clean_input = preprocess_text(user_input)
        seq = tokenizer.texts_to_sequences([clean_input])
        padded = pad_sequences(seq, maxlen=MAXLEN, padding='post')
        prediction = model.predict(padded)[0][0]
        if prediction > 0.5:
            st.success(f"😊 Positive Sentiment (Confidence: {prediction*100:.2f}%)")
        else:
            st.error(f"😡 Negative Sentiment (Confidence: {(1-prediction)*100:.2f}%)")
    else:
        st.warning("Please enter some text first!")
# ...existing code...
