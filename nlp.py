import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
import re
import pickle
import os
import requests
from zipfile import ZipFile

# Function to download and extract model files
def download_and_extract_model():
    # URLs for the ZIP parts on GitHub (replace with your actual URLs)
    url_part1 = 'https://github.com/ahmadarif238/Sentiment-Analysis-NLP-model-using-the-IMDB-dataset/blob/7b8d2553032dd2294db4c9d52bb32642101d2b55/trained_model.part1.rar'
    url_part2 = 'https://github.com/ahmadarif238/Sentiment-Analysis-NLP-model-using-the-IMDB-dataset/blob/e3c6bd31201f05dd923ca11c7d67d2ad882b6d33/trained_model.part2.rar'

    # Download the ZIP parts
    for url, filename in [(url_part1, 'trained_model.part1.zip'), (url_part2, 'trained_model.part2.zip')]:
        response = requests.get(url)
        with open(filename, 'wb') as file:
            file.write(response.content)

    # Combine and extract the ZIP files
    combined_zip_path = 'combined_trained_model.zip'
    with open(combined_zip_path, 'wb') as output_file:
        for part in ['trained_model.part1.zip', 'trained_model.part2.zip']:
            with open(part, 'rb') as input_file:
                output_file.write(input_file.read())

    # Extract the combined ZIP file
    with ZipFile(combined_zip_path, 'r') as zip_ref:
        zip_ref.extractall()

    # Clean up the individual parts and the combined ZIP file
    os.remove('trained_model.part1.zip')
    os.remove('trained_model.part2.zip')
    os.remove(combined_zip_path)

# Call the function to download, combine, and extract the model file
download_and_extract_model()

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
