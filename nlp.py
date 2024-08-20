import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
import re
import pickle
import os

# Function to download and combine split files
def download_and_combine_files():
    part1_url = "https://github.com/ahmadarif238/Sentiment-Analysis-NLP-model-using-the-IMDB-dataset/raw/main/trained_model.part1.rar"
    part2_url = "https://github.com/ahmadarif238/Sentiment-Analysis-NLP-model-using-the-IMDB-dataset/raw/main/trained_model.part2.rar"
    
    # Download the files
    os.system(f"curl -L {part1_url} -o trained_model.part1.rar")
    os.system(f"curl -L {part2_url} -o trained_model.part2.rar")
    
    # Combine the files
    os.system("cat trained_model.part1.rar trained_model.part2.rar > trained_model_combined.rar")
    
    # Extract the combined file
    os.system("unrar x trained_model_combined.rar")
    
    # Check if the extraction was successful
    if not os.path.exists('trained_model.pkl'):
        st.error("Failed to extract the trained_model.pkl file. Please check the extraction process.")
        st.stop()

# Call the function to download and combine files
download_and_combine_files()

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
