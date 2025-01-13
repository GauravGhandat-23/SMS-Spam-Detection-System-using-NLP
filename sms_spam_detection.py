# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
import re
import zipfile
import os
import requests

# Download NLTK stopwords
nltk.download('stopwords')

# Function to preprocess the text data
def preprocess_text(text):
    # Remove non-alphabetical characters and convert to lower case
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
    return text

# Load the dataset
def load_data():
    # URL of the ZIP file containing the dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    
    # Temporary directory to extract the ZIP file
    temp_dir = "smsspamcollection"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # Download the ZIP file and extract it
    with open(f"{temp_dir}/smsspamcollection.zip", "wb") as f:
        f.write(requests.get(url).content)

    # Extract the ZIP file
    with zipfile.ZipFile(f"{temp_dir}/smsspamcollection.zip", "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    # Load the dataset (SMSSpamCollection) from the extracted files
    dataset = pd.read_csv(f"{temp_dir}/SMSSpamCollection", sep='\t', header=None)
    
    # Convert column names to string and strip any leading/trailing spaces
    dataset.columns = dataset.columns.astype(str).map(str.strip)
    
    # Rename columns to 'Label' and 'Message'
    dataset.columns = ['Label', 'Message']
    
    return dataset

# Preprocess the dataset
def preprocess_data(dataset):
    # Apply the preprocessing function to the 'Message' column
    dataset['Message'] = dataset['Message'].apply(preprocess_text)
    return dataset

# Split the data into training and testing sets
def split_data(dataset):
    X = dataset['Message']
    y = dataset['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# TF-IDF Vectorization
def vectorize_data(X_train, X_test):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, vectorizer

# Train the Naive Bayes model
def train_model(X_train_tfidf, y_train):
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test_tfidf, y_test):
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Streamlit UI
def main():
    st.title("SMS Spam Detection")
    
    # Load the default dataset
    dataset = load_data()
    
    # Show a snippet of the dataset
    st.write("### Dataset Preview:")
    st.dataframe(dataset.head())

    # Preprocess the data
    dataset = preprocess_data(dataset)
    
    # Split the data
    X_train, X_test, y_train, y_test = split_data(dataset)
    
    # Vectorize the data
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize_data(X_train, X_test)
    
    # Train the model
    model = train_model(X_train_tfidf, y_train)
    
    # Evaluate the model
    accuracy = evaluate_model(model, X_test_tfidf, y_test)
    st.write(f"### Model Accuracy: {accuracy * 100:.2f}%")
    
    # Input field for new SMS message
    user_input = st.text_area("Enter your SMS message here:")
    
    if st.button('Classify'):
        if user_input:
            processed_input = preprocess_text(user_input)
            input_tfidf = vectorizer.transform([processed_input])
            prediction = model.predict(input_tfidf)[0]
            
            if prediction == 'spam':
                st.write("This message is **SPAM**")
            else:
                st.write("This message is **HAM**")
        else:
            st.write("Please enter an SMS message to classify.")

if __name__ == "__main__":
    main()
