import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
import re
import pickle
import os
import requests
from zipfile import ZipFile, BadZipFile

# Function to download a file from a URL
def download_file(url, local_filename):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        st.error(f"Failed to download {local_filename}. Status code: {response.status_code}")
        st.stop()

# Function to download and extract the model
def download_and_extract_model():
    # Download the parts
    part1_url = "https://github.com/ahmadarif238/Sentiment-Analysis-NLP-model-using-the-IMDB-dataset/raw/4f2f2b376857a1171fbf41f962bc394b8c16f9fd/trained_model.part1.zip"
    part2_url = "https://github.com/ahmadarif238/Sentiment-Analysis-NLP-model-using-the-IMDB-dataset/raw/4f2f2b376857a1171fbf41f962bc394b8c16f9fd/trained_model.part2.zip"
    
    download_file(part1_url, 'trained_model.part1.zip')
    download_file(part2_url, 'trained_model.part2.zip')
    
    # Combine the parts
    combined_zip_path = 'trained_model_combined.zip'
    with open(combined_zip_path, 'wb') as combined:
        with open('trained_model.part1.zip', 'rb') as part1:
            combined.write(part1.read())
        with open('trained_model.part2.zip', 'rb') as part2:
            combined.write(part2.read())
    
    # Extract the model from the combined zip file
    try:
        with ZipFile(combined_zip_path, 'r') as zip_ref:
            zip_ref.extractall()
    except BadZipFile:
        st.error("The combined file is not a valid ZIP file. Please check the file.")
        st.stop()

# Call the function to download and extract the model
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
