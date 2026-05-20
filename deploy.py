import streamlit as st
import pickle
import sklearn
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

tfidf = pickle.load(open('vectorizer2.pkl', 'rb'))
model = pickle.load(open('model2.pkl', 'rb'))

st.title('EMAIL SPAM CLASSIFIER')
input_mail = st.text_area('ENTER THE EMAIL')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

if st.button('Predict'):
    transformed_mail = transform_text(input_mail)
    vectrozered_mail = tfidf.transform([transformed_mail])
    pred = model.predict(vectrozered_mail)[0]
    if pred == 1:
        st.header("spam")
    else:
        st.header("ham")