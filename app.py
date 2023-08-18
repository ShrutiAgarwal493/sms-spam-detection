import streamlit as st
import streamlit.components.v1 as com
import pickle
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

html_string_spam = """
<style>
.alert-danger {
    color: #721c24;
    background-color: #f8d7da;
    border-color: #f5c6cb;
    height: 100px;
    margin-bottom:20px;
    border-radius:20px;
}
</style>
<div class="alert alert-danger" role="alert">
    <center><h2><b>Wait a minute! Don't trust on there is a SPAM! <img src="https://img.icons8.com/officel/32/000000/surprised.png"/></b></h2></center>
</div>
"""

html_string_ham = """
<style>
.alert-success {
    color: #155724;
    background-color: #d4edda;
    border-color: #c3e6cb;
    height: 100px;
    margin-bottom:20px;
    border-radius:20px;
}
</style>
<div class="alert alert-success" role="alert">
    <center><h2><b>Thank God!, this is a normal message.<img src="https://img.icons8.com/emoji/48/000000/grinning-face-emoji.png"/></b></h2></center>
</div>
"""

def main():
    com.html("""<header style="background-color: #333; padding: 20px; text-align: center;">
          <h1 style="color: white; margin: 0;">SMS SPAM CLASSIFIER</h1>
          <p style="color: #e6af2e; margin: 5px 0 0 0;">Detect Spam Messages with Machine Learning</p>
        </header>""")

    with st.form("text_area_form"):
        input_sms = st.text_area("Enter your SMS", "")
        submitted = st.form_submit_button("Predict")

        if submitted:
            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]
            
            if input_sms.strip() == "":
                st.error("Text area cannot be empty!")
            else:
                if result == 1:
                    st.markdown(html_string_spam, unsafe_allow_html=True)
                else:
                    st.markdown(html_string_ham, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

com.html(
    """
<footer style="background-color: #333; padding: 20px; color: white; text-align: center;">
  <p>Made with ❤️ by <a href="#" style="color: #e6af2e; text-decoration: none;" rel="noopener noreferrer" target="_blank">Shruti Agarwal</a></p>
  <p>Follow me on: 
    <a href="#" style="color: #e6af2e; text-decoration: none; margin: 0 10px;" rel="noopener noreferrer" target="_blank">Twitter</a>
    <a href="#" style="color: #e6af2e; text-decoration: none; margin: 0 10px;" rel="noopener noreferrer" target="_blank">LinkedIn</a>
    <a href="#" style="color: #e6af2e; text-decoration: none; margin: 0 10px;" rel="noopener noreferrer" target="_blank">GitHub</a>
  </p>
</footer>
</div>
"""
)
