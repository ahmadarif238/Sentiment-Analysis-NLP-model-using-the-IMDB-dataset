import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
import re
import pickle
import requests
import os

# Function to download the model file from Google Drive
def download_model():
    model_url = "https://drive.google.com/uc?export=download&id=1hd-BRqA_t3E7HH1k0nkTYTfuPApw3LwS"
    response = requests.get(model_url, stream=True)
    
    # Check if the download was successful
    if response.status_code == 200:
        with open('trained_model.pkl', 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        st.error("Failed to download the trained_model.pkl file. Please check the download process.")
        st.stop()

# Call the function to download the model file
download_model()

# Load the trained model and vectorizer
with open('trained_model.pkl', 'rb') as f:
    ensemble_model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

nltk.download('punkt')

def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    tokens = word_tokenize(text)  # Tokenize the text
    return ' '.join(tokens)  # Join tokens back into a single string

st.title("Movie Review Sentiment Analysis")

# Function to get user input and predict sentiment
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
                # Preprocess the review
                review_preprocessed = preprocess_text(review)
                # Transform the review using the TF-IDF vectorizer
                review_tfidf = vectorizer.transform([review_preprocessed])
                # Predict the sentiment
                prediction = ensemble_model.predict(review_tfidf)
                sentiment = "Positive" if prediction[0] == 'positive' else "Negative"
                st.write(f"The sentiment of your review is: {sentiment}")
                if st.button("Submit Another Review"):
                    st.experimental_rerun()
            else:
                st.write("Please enter a review.")

get_review()
