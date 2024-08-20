import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
import re
import pickle
import requests
from io import BytesIO

# Function to download the model file from Google Drive
def download_model():
    model_url = "https://drive.google.com/uc?export=download&id=1hd-BRqA_t3E7HH1k0nkTYTfuPApw3LwS"
    response = requests.get(model_url, stream=True)
    
    if response.status_code == 200:
        return BytesIO(response.content)
    else:
        st.error("Failed to download the trained_model.pkl file.")
        st.stop()

# Load the trained model and vectorizer
try:
    model_data = download_model()
    with BytesIO(model_data.read()) as f:
        ensemble_model = pickle.load(f)
except pickle.UnpicklingError as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Assuming you have a vectorizer.pkl file, replace with actual path
try:
    vectorizer_data = download_model()
    with BytesIO(vectorizer_data.read()) as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    st.error("vectorizer.pkl file not found.")
    st.stop()
except pickle.UnpicklingError as e:
    st.error(f"Error loading vectorizer: {e}")
    st.stop()

nltk.download('punkt')

def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    tokens = word_tokenize(text)  # Tokenize the text
    return ' '.join(tokens)  # Join tokens back into a single string

st.title("Movie Review Sentiment Analysis")

def get_review():
    movie_suggestions = [
        "Borderlands (2024)",
        "It Ends With Us (2024)",
        "Deadpool & Wolverine (2024)",
        "Trap (2024)",
        "Cuckoo (2024)"
    ]
    
    st.write("Here are some movie suggestions for you:")
    for movie in movie_suggestions:
        st.write(f"- {movie}")

    movie_name = st.text_input("Enter the movie name:")
    if movie_name:
        review = st.text_area("Enter your review for the movie:")
        if st.button("Submit Review"):
            if review:
                review_preprocessed = preprocess_text(review)
                review_tfidf = vectorizer.transform([review_preprocessed])
                prediction = ensemble_model.predict(review_tfidf)
                sentiment = "Positive" if prediction[0] == 'positive' else "Negative"
                st.write(f"The sentiment of your review is: {sentiment}")
                if st.button("Submit Another Review"):
                    st.experimental_rerun()
            else:
                st.write("Please enter a review.")

get_review()
