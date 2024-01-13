import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re


def predict_bullying_type(input_text, bad_words):
    df = pd.read_excel("Cleaned_Hinglish_dataset.xlsx")
    df.dropna(subset=["cleaned_text"], inplace=True)
    X = df["cleaned_text"]
    y = df["type"]
    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(X)
    logistic_regression_classifier = LogisticRegression(max_iter=1000)
    logistic_regression_classifier.fit(X_tfidf, y)

    text_tfidf = tfidf_vectorizer.transform([input_text])
    predicted_types = logistic_regression_classifier.predict(text_tfidf)

    detected_bad_words = []

    for bad_word in bad_words:
        if re.search(r'\b{}\b'.format(re.escape(bad_word)), input_text, flags=re.IGNORECASE):
            detected_bad_words.append(bad_word)
        input_text = re.sub(r'\b{}\b'.format(re.escape(bad_word)), '*******', input_text, flags=re.IGNORECASE)

    prediction = "Not Cyberbullying" if predicted_types[0] != "OAG" else "Cyberbullying"
    return prediction, predicted_types[0], input_text, detected_bad_words

st.title("Cyberbullying Detection App (Hinglish)")

input_text = st.text_area("Enter a text for cyberbullying detection:")


if st.button("Predict"):
    bad_words = ['bahenchod', 'peshan', 'behenchod', 'bhenchod', 'bhenchodd', 'b.c.', 'bc', 'bakchod', 'bakchodd', 'bakchodi', 'bevda', 'bewda', 'bevdey', 'bewday', 'bhadwaa', 'bhosada', 'bhosda', 'bhosdaa', 'bhosdike', 'bhonsdike', 'bhosdiki', 'bhosdiwala', 'bhosdiwale', 'Bhosadchodal', 'Bhosadchod', 'Bhosadchodal', 'Bhosadchod', 'babbe', 'babbey', 'bube', 'bubey', 'bur', 'burr', 'buurr', 'buur', 'charsi', 'chooche', 'choochi', 'chuchi', 'chhod', 'chod', 'chodd', 'chudne', 'chudney', 'chudwa', 'chudwaa', 'chudwane', 'chudwaane', 'chaat', 'choot', 'chut', 'chute', 'chutia', 'chutiya', 'chutiye', 'dalaal', 'dalal', 'dalle', 'dalley', 'fattu', 'gadha', 'gadhe', 'gadhalund', 'gaand', 'gand', 'gandu', 'gandfat', 'gandfut', 'gandiya', 'gandiye', 'goo', 'gu', 'gote', 'gotey', 'gotte', 'hag', 'haggu', 'hagne', 'hagney', 'harami', 'haramjada', 'haraamjaada', 'haramzyada', 'haraamzyaada', 'haraamjaade', 'haraamzaade', 'haraamkhor', 'haramkhor', 'jhat', 'jhaat', 'jhaatu', 'jhatu', 'kutta', 'kutte', 'kuttey', 'kutia', 'kutiya', 'kuttiya', 'kutti',  'landi', 'landy', 'laude', 'laudey', 'laura', 'lora', 'lauda', 'ling', 'loda', 'lode', 'lund', 'launda', 'lounde', 'laundey', 'laundi', 'loundi', 'laundiya', 'loundiya', 'lulli', 'maar ja', 'madarchod', 'madarchodd', 'madarchood', 'madarchoot', 'madarchut', 'm.c.', 'mc', 'mamme', 'mammey', 'moot', 'mut', 'mootne', 'mutne', 'mooth', 'muth', 'nunnu', 'pesaab', 'pesab', 'peshaab', 'peshab', 'pillay', 'pille', 'pilley', 'pisaab', 'pisab', 'porkistan', 'raand', 'rand', 'randi', 'randy', 'suar', 'tatti', 'tatty', 'ullu', 'pappu']
    prediction, bullying_type, filtered_text, detected_bad_words = predict_bullying_type(input_text, bad_words)
    
    st.write("Prediction:", prediction)
    st.write("Cyberbullying Type: ", bullying_type)
    if detected_bad_words:
        st.write("Bad Words:", ', '.join(detected_bad_words))
    else:
        st.write("<span style='color:cyan;'>No bad words found.</span>", unsafe_allow_html=True)
    
    if bad_words:
        st.write("Filtered Text:")
        st.write(f"<span style='color:red; font-weight:bold'>{filtered_text}</span>", unsafe_allow_html=True) 
    else:
        st.write("Original Text:")
        st.write(f"{filtered_text}", unsafe_allow_html=True)
    
st.header("Sample Texts")
st.write("es" + "<span style='color:red; font-weight:bold'> kutte</span> ko jail daal desh drohi hai", unsafe_allow_html=True)
st.write("" + "<span style='color:red; font-weight:bold'>pappu </span>gandhi", unsafe_allow_html=True)
st.write("waise bandhu jet lag se bachne ke liye raat ko baje ke baad jao")
st.write("news main rhane ka accha mauka hai desh aur army army aur bharat mata")
