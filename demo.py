import streamlit as st
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model
model = joblib.load('model.pkl')

def preprocess(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    return text

st.title('SMS Spam Classifier')

# Input text box for user to enter SMS message
input_text = st.text_area('Enter an SMS message:', '')

if st.button('Predict'):
    # Preprocess the input text
    preprocessed_text = preprocess(input_text)
    
    # Make prediction using the model
    prediction = model.predict([preprocessed_text])[0]
    
    # Display the prediction result
    if prediction == 'spam':
        st.error('This is a SPAM message!')
    else:
        st.success('This is a HAM message.')
