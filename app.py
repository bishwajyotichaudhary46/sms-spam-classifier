import streamlit as st
import pickle
import nltk
import sklearn
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.models import load_model
from keras.utils import pad_sequences

ps = PorterStemmer()

def transform_text(text):
  text  = text.lower()
  text = nltk.word_tokenize(text)
  y =[]
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



tfidf = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
tokenizer = pickle.load(open('tokenizer.pkl','rb'))
lstm_model = load_model("lstm_model.h5")

st.title("Email/Spam Classifier")

input_sms = st.text_input("Enter the message")





st.title("Prediction Through ML")
button1 = st.button("Predict by ML")
if button1:

  transform_sms = transform_text(input_sms)

  vector_input = tfidf.transform([transform_sms]).toarray()

  result  = model.predict(vector_input)[0]

  if result == 1:
    st.header("Spam")

  else:
    st.header("Not Spam")

st.title("Prediction Through RNN_LSTM")
button2 = st.button("Button by LSTM")
if button2:
  max_len = 150
  transform_sms = transform_text(input_sms)

  txt_sms = tokenizer.texts_to_sequences(transform_sms)
  vector_input = pad_sequences(txt_sms, maxlen=max_len)

  result = lstm_model.predict(vector_input)[0]


  if result ==1:
    st.header("Spam")

  else:
    st.header("Not Spam")

