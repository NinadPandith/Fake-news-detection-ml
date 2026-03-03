import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle
import joblib
import os

def main():
    print("Starting Fake News Detection Training...")
    
    # 1. Load Datasets
    try:
        if not os.path.exists('Fake.csv') or not os.path.exists('True.csv'):
            print("Error: 'Fake.csv' and/or 'True.csv' not found.")
            print("Please ensure you have both dataset files in this directory.")
            return

        fake_df = pd.read_csv('Fake.csv')
        true_df = pd.read_csv('True.csv')
        print("Datasets loaded successfully.")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    # 2. Add labels (Fake=0, Real=1)
    fake_df['label'] = 0
    true_df['label'] = 1

    # 3. Combine datasets
    df = pd.concat([fake_df, true_df], ignore_index=True)
    
    # Check for 'text' column
    if 'text' not in df.columns:
        print("Error: The datasets must contain a 'text' column.")
        return
        
    # 4. Shuffle dataset
    print("Shuffling dataset...")
    df = shuffle(df, random_state=42).reset_index(drop=True)

    # Extract features and labels
    X = df['text']
    y = df['label']
    
    # Print number of samples and distribution
    print(f"\nTotal number of samples: {len(df)}")
    print("Label distribution:")
    print(y.value_counts().rename(index={1: 'Real (1)', 0: 'Fake (0)'}))

    # Deal with excessively small datasets gracefully to avoid stratify failure
    if len(df) < 5:
        print("\nWarning: Dataset is too small for meaningful training!")
        stratify_param = None
    else:
        stratify_param = y

    # 5. Split data 80/20 with stratification
    print("\nSplitting data into 80% training and 20% testing...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_param)
    except ValueError as e:
        print(f"\nWarning: Stratified split failed (usually due to small dataset size): {e}")
        print("Falling back to unstratified split.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. Use TF-IDF vectorizer
    print("Applying TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 7. Train Logistic Regression
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Only print confusion matrix/classification report if we have a reasonable test set
    if len(y_test) > 1:
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        # Suppress zero_division warnings during tiny mock dataset evaluation
        print(classification_report(y_test, y_pred, zero_division=0))
    else:
        print(f"\nTest set too small for confusion matrix ({len(y_test)} sample).")
        print(f"Actual label: {y_test.iloc[0]}, Predicted: {y_pred[0]}")

    # 8. Save model and vectorizer
    print("\nSaving model and vectorizer...")
    joblib.dump(model, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    
    # Print confirmation message
    print("\nTraining Complete! Successfully saved as 'model.pkl' and 'vectorizer.pkl'.")

if __name__ == "__main__":
    main()
