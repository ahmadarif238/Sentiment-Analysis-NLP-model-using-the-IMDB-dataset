import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
import re
import pickle
import requests
from io import BytesIO

# Function to download the file from Google Drive
def download_file_from_google_drive(file_id):
    base_url = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    
    response = session.get(f"{base_url}&id={file_id}", stream=True)
    if response.status_code == 200:
        return BytesIO(response.content)
    else:
        st.error("Failed to download the file from Google Drive.")
        st.stop()

# Function to load a pickle file from Google Drive
def load_pickle_from_google_drive(file_id):
    try:
        file_data = download_file_from_google_drive(file_id)
        with BytesIO(file_data.read()) as f:
            return pickle.load(f)
    except pickle.UnpicklingError as e:
        st.error(f"Error loading pickle file: {e}")
        st.stop()

# File ID for the trained_model.pkl file
model_file_id = "1hd-BRqA_t3E7HH1k0nkTYTfuPApw3LwS"

# Download and load the model
ensemble_model = load_pickle_from_google_drive(model_file_id)

# Load the vectorizer if it's also on Google Drive
# If you don't have a vectorizer file, you can comment out these lines
# vectorizer_file_id = "your_vectorizer_file_id"
# vectorizer = load_pickle_from_google_drive(vectorizer_file_id)

# If you need a local vectorizer file, provide its path accordingly

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
                # Ensure vectorizer is properly loaded; use local path if necessary
                try:
                    review_tfidf = vectorizer.transform([review_preprocessed])
                    prediction = ensemble_model.predict(review_tfidf)
                    sentiment = "Positive" if prediction[0] == 'positive' else "Negative"
                    st.write(f"The sentiment of your review is: {sentiment}")
                except Exception as e:
                    st.error(f"Error processing the review: {e}")
                
                if st.button("Submit Another Review"):
                    st.experimental_rerun()
            else:
                st.write("Please enter a review.")

get_review()
