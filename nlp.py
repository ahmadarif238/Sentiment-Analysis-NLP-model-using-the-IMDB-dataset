import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
import re
import pickle
import os
import requests
import hashlib

# Function to verify file integrity using checksums
def verify_checksum(file_path, expected_checksum):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest() == expected_checksum

# Function to download and combine split files
def download_and_combine_files():
    part1_url = "https://github.com/ahmadarif238/Sentiment-Analysis-NLP-model-using-the-IMDB-dataset/raw/main/trained_model.part1.rar"
    part2_url = "https://github.com/ahmadarif238/Sentiment-Analysis-NLP-model-using-the-IMDB-dataset/raw/main/trained_model.part2.rar"
    part1_checksum = "expected_checksum_part1"  # Replace with actual checksum
    part2_checksum = "expected_checksum_part2"  # Replace with actual checksum
    
    # Download the files
    try:
        with open("trained_model.part1.rar", "wb") as f:
            f.write(requests.get(part1_url).content)
        with open("trained_model.part2.rar", "wb") as f:
            f.write(requests.get(part2_url).content)
    except Exception as e:
        st.error(f"Error downloading files: {e}")
        return False
    
    # Verify file integrity
    if not verify_checksum("trained_model.part1.rar", part1_checksum):
        st.error("Checksum verification failed for part1.rar")
        return False
    if not verify_checksum("trained_model.part2.rar", part2_checksum):
        st.error("Checksum verification failed for part2.rar")
        return False
    
    # Combine the files
    try:
        with open("trained_model_combined.rar", "wb") as combined:
            for part in ["trained_model.part1.rar", "trained_model.part2.rar"]:
                with open(part, "rb") as f:
                    combined.write(f.read())
    except Exception as e:
        st.error(f"Error combining files: {e}")
        return False
    
    # Extract the combined file using p7zip
    try:
        os.system("7z x trained_model_combined.rar")
    except Exception as e:
        st.error(f"Error extracting files: {e}")
        return False
    
    return True

# Call the function to download and combine files
if download_and_combine_files():
    # Load the trained model and vectorizer
    try:
        with open('trained_model.pkl', 'rb') as f:
            ensemble_model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
    except FileNotFoundError:
        st.error("Model files not found. Please check the file paths.")
    except Exception as e:
        st.error(f"Error loading model files: {e}")
else:
    st.error("Failed to download and combine model files.")

# Download the 'punkt' tokenizer
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
