import joblib
import string
import re
import numpy as np

def predict_news(text):
    """
    Preprocess the input text, vectorize it, and predict using the loaded model.
    Returns:
      - prediction_label ('Real' or 'Fake')
      - fake probability
      - real probability
      - top_fake_words: list of dicts with words contributing to 'Fake'
    """
    # Load model and vectorizer
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    # Preprocess input text (lowercase, remove punctuation)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()

    # Transform text using loaded vectorizer
    text_vectorized = vectorizer.transform([text])

    # Predict using loaded model
    prediction = model.predict(text_vectorized)[0]
    
    # Get probabilities
    probabilities = model.predict_proba(text_vectorized)[0]

    # Calculate confidence score in percentage
    fake_prob = probabilities[0] * 100
    real_prob = probabilities[1] * 100

    # Prediction label
    # 0 = Fake, 1 = Real
    prediction_label = "Real" if prediction == 1 else "Fake"

    # -- Explainability: Extract top words influencing the prediction --
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    
    # Get indices of words present in the input text
    non_zero_indices = text_vectorized[0].nonzero()[1]
    
    contributions = []
    for idx in non_zero_indices:
        word = feature_names[idx]
        tfidf_val = text_vectorized[0, idx]
        coef = coefficients[idx]
        
        # Contribution to the decision function (coef * feature_value)
        # Since class 1 is Real and class 0 is Fake:
        # Negative contribution pushes towards Fake
        contribution = tfidf_val * coef
        contributions.append((word, contribution))
        
    # Sort by contribution (most negative first = strongest Fake indicators)
    contributions_sorted = sorted(contributions, key=lambda x: x[1])
    
    top_fake_words = []
    for word, contrib in contributions_sorted:
        if contrib < 0:
            top_fake_words.append({"Word": word, "Fake Signal Weight": round(-contrib, 4)})
        
        # Limit to top 10 words
        if len(top_fake_words) >= 10:
            break

    return prediction_label, fake_prob, real_prob, top_fake_words

if __name__ == "__main__":
    # Example usage
    sample_news = "The economy is booming according to the newly released statistics by the government."
    
    try:
        label, fake_p, real_p, top_fake = predict_news(sample_news)
        print(f"Text: {sample_news}")
        print(f"Prediction: {label}")
        print(f"Fake Probability: {fake_p:.2f}%")
        print(f"Real Probability: {real_p:.2f}%")
        print(f"Top Fake Words: {top_fake}")
    except Exception as e:
        print(f"An error occurred: {e}")
