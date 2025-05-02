import streamlit as st
import pickle
import fitz  # PyMuPDF
import spacy

# Load model and vectorizer
model = pickle.load(open("model/resume_model.pkl", "rb"))

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# PDF Parser
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Keyword extraction (basic)
def extract_keywords(text):
    doc = nlp(text)
    keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN", "ADJ"] and not token.is_stop]
    return list(set(keywords))[:20]

# App UI
st.set_page_config(page_title="Resume Classifier AI", layout="wide")
st.title("🧠 AI Resume Classifier + Insights")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)
    st.subheader("📄 Extracted Resume Text")
    st.text_area("Text", text[:2000], height=300)

    if st.button("🔍 Classify Resume"):
        pred = model.predict([text])[0]
        st.success(f"✅ Predicted Category: **{pred}**")

        # Show Keywords
        st.subheader("🔑 Top Keywords from Resume")
        keywords = extract_keywords(text)
        st.write(", ".join(keywords))

from wordcloud import WordCloud
import matplotlib.pyplot as plt

def generate_wordcloud(text):
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wc

if st.button("📊 Generate Word Cloud"):
    st.subheader("🖼 Word Cloud of Resume Keywords")
    wc = generate_wordcloud(text)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)
