import streamlit as st
import joblib

# Set page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Load the trained machine learning pipeline
@st.cache_resource
def load_model():
    # Ensure 'fake_news_model.pkl' is in the same directory
    return joblib.load("fake_news_model.pkl")

model = load_model()

# App UI Header
st.title("📰 Fake News Detection System")
st.write(
    "Enter the news article headline and text below to check whether it is predicted "
    "to be **Real News** or **Fake News**."
)

# Input fields
title_input = st.text_input("News Title / Headline", placeholder="Enter the headline here...")
text_input = st.text_area("News Article Body", placeholder="Enter the full text or body of the article here...", height=200)

# Prediction button
if st.button("Detect News Type", type="primary"):
    if not title_input.strip() and not text_input.strip():
        st.warning("Please enter at least a title or text to analyze.")
    else:
        # Combine inputs in the same way as training: title + " " + text
        combined_content = f"{title_input} {text_input}"
        
        # Make prediction
        prediction = model.predict([combined_content])[0]
        
        # Display results based on label encoding (0 = Real News, 1 = Fake News)
        st.markdown("---")
        st.subheader("Prediction Result:")
        
        if prediction == 1:
            st.error("🚨 **Prediction: Fake News** (Caution: This article exhibits characteristics commonly found in fake news stories.)")
        else:
            st.success("✅ **Prediction: Real News** (This article appears to be consistent with real news reporting.)")

# Footer info
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Built with Streamlit & Scikit-Learn</p>", unsafe_allow_html=True)