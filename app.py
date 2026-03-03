import streamlit as st
import pandas as pd
import joblib
import warnings
import time
warnings.filterwarnings('ignore')

try:
    from predict import predict_news
    DEPENDENCY_MISSING = False
    ERROR_MSG = ""
except ImportError as e:
    DEPENDENCY_MISSING = True
    ERROR_MSG = e

# Streamlit Page Configuration
st.set_page_config(
    page_title="AI Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        margin-bottom: 20px;
        text-align: center;
    }
    .fake-news {
        background-color: #ffcccc;
        color: #cc0000;
        border: 2px solid #cc0000;
    }
    .real-news {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #155724;
    }
    .stDataFrame {
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

def main():
    if DEPENDENCY_MISSING:
        st.error(f"Failed to load dependencies: `{ERROR_MSG}`. Please ensure predict.py and model files are in the same folder.")
        return

    # Header section
    st.title("📰 AI Fake News Detection App")
    st.markdown("Enter a news article or snippet below, and our Machine Learning model will predict whether it's **Real** or **Fake**, along with confidence probabilities and an explainability breakdown.")
    
    # Text Input
    user_input = st.text_area("News Article:", height=200, placeholder="Paste the news text or article here...")
    
    # Check button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submit_button = st.button("Check News", use_container_width=True, type="primary")
    
    st.markdown("---")

    if submit_button:
        cleaned_input = user_input.strip()
        
        if not cleaned_input:
             st.warning("⚠️ Please enter some news text to analyze.")
        else:
            # Loading Spinner
            with st.spinner("Analyzing text using NLP & Logistic Regression..."):
                time.sleep(0.5) 
                
                try:
                    # Call prediction (Now returns top fake words as well)
                    prediction_label, fake_prob, real_prob, top_fake_words = predict_news(cleaned_input)
                    
                    # 1. Display Primary Result visually
                    if prediction_label == "Real":
                        st.markdown(f'<div class="result-box real-news"><h2>✅ The Model Predicts: REAL NEWS</h2></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="result-box fake-news"><h2>🚨 The Model Predicts: FAKE NEWS</h2></div>', unsafe_allow_html=True)
                        
                    # 2. Show Probabilities with Progress Bars
                    st.subheader("Confidence Breakdown")
                    prog_col1, prog_col2 = st.columns(2)
                    
                    with prog_col1:
                        st.markdown(f"**Real ({real_prob:.1f}%)**")
                        st.progress(int(real_prob) if real_prob > 0 else 0)
                        
                    with prog_col2:
                        st.markdown(f"**Fake ({fake_prob:.1f}%)**")
                        st.progress(int(fake_prob) if fake_prob > 0 else 0)
                        
                    # 3. Explainability: Show Top Words contributing to Fake Prediction
                    if top_fake_words:
                        st.markdown("### 🔍 Explainability: Top 'Fake' Words")
                        st.markdown("These are the words from your text that pushed the model the most towards a 'Fake' prediction, ranked by their signal weight.")
                        
                        # Convert to DataFrame for clean table display
                        df_words = pd.DataFrame(top_fake_words)
                        df_words.index = df_words.index + 1 # 1-based indexing
                        st.table(df_words)
                    else:
                        st.info("No significant 'Fake' trigger words were found in this text.")

                except Exception as ex:
                    st.error(f"An error occurred during prediction: {ex}. Make sure the model is trained.")
    
    # Footer Section
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray; font-size: small;'>Built by Ninad using Machine Learning & Streamlit</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
