import nltk
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#transform text into lower case,special characters

ps = PorterStemmer()
nltk.download('stopwords')
stopwords.words('english')

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
    if i not in stopwords.words('english') and i not in string.punctuation :
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    y.append(ps.stem(i))
  return " ".join(y)

model = pickle.load(open('AB-model.pkl','rb'))
st.title('Email-Classifier')

input_text = st.text_input("Enter the Mail")

if st.button('predict'):

    transformed_text = transform_text(input_text)

    vector_input = model.transform([transformed_text])

    result = model.predict(vector_input)

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")