import streamlit as st
import pickle
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Nettoyage du texte
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    return text

# Charger modèle et vectoriseur
model = pickle.load(open("model_svm.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

st.title('Détection de Spam')

user_input = st.text_input("Entrez le message :")

if st.button('Prédire'):
    user_input_clean = clean_text(user_input)
    user_vect = tfidf.transform([user_input_clean])
    prediction = model.predict(user_vect)
    if prediction[0] == 1:
        st.error("C'est un SPAM 🚫")
    else:
        st.success("Ce n'est pas un spam ✅")