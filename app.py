import streamlit as st
import pickle
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

# Function to transform text
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
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('AB-model.pkl','rb'))

# Set page title and header
st.set_page_config(page_title="Email Classifier", page_icon=":shield:", layout="centered")

# Create a header
st.title("Email Classifier")

# Input for user text
input_email = st.text_area("Enter the email", "Type your email content here...")

# Predict button
if st.button('Predict'):
    # Check if input is empty
    if not input_email.strip():
        st.warning("Please enter an email for classification.")
    else:
        # Preprocess and predict
        transformed_email = transform_text(input_email)
        vector_input = tfidf.transform([transformed_email])
        result = model.predict(vector_input)[0]

        # Display result
        if result == 1:
            st.success("Prediction: Spam :warning:")
        else:
            st.success("Prediction: Not Spam :thumbsup:")
