import streamlit as st
import pickle
import re

model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review and I'll tell you if it's positive or negative!")

review = st.text_area("Enter your review here:")

if st.button("Analyze"):
    if review:
        cleaned = clean_text(review)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)
        if prediction[0] == 1:
            st.success("Positive Review 😊")
        else:
            st.error("Negative Review 😞")
    else:
        st.warning("Please enter a review first!")