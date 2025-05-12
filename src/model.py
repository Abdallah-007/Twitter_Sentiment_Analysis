"""
Twitter Sentiment Analysis - Model Training
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import local preprocessing module
from preprocess import load_and_preprocess_data

def vectorize_text(train_texts, val_texts=None, method='tfidf', **kwargs):
    """
    Convert text data to numerical features using TF-IDF or Bag-of-Words
    
    Args:
        train_texts (list): List of training text samples
        val_texts (list): List of validation text samples
        method (str): Vectorization method ('tfidf' or 'bow')
        **kwargs: Additional arguments for the vectorizer
    
    Returns:
        tuple: (train_features, val_features, vectorizer)
    """
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(**kwargs)
    elif method == 'bow':
        vectorizer = CountVectorizer(**kwargs)
    else:
        raise ValueError("Method must be either 'tfidf' or 'bow'")
    
    train_features = vectorizer.fit_transform(train_texts)
    
    val_features = None
    if val_texts is not None:
        val_features = vectorizer.transform(val_texts)
    
    return train_features, val_features, vectorizer

def train_model(X_train, y_train, model_type='nb', **kwargs):
    """
    Train a sentiment analysis model
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_type (str): Type of model to train ('nb', 'lr', 'svm', 'rf')
        **kwargs: Additional arguments for the model
    
    Returns:
        Trained model
    """
    if model_type == 'nb':
        model = MultinomialNB(**kwargs)
    elif model_type == 'lr':
        model = LogisticRegression(**kwargs)
    elif model_type == 'svm':
        model = LinearSVC(**kwargs)
    elif model_type == 'rf':
        model = RandomForestClassifier(**kwargs)
    else:
        raise ValueError("Model type must be 'nb', 'lr', 'svm', or 'rf'")
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, labels=None):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        labels: List of label names
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix,
        'y_pred': y_pred
    }

def save_model(model, vectorizer, model_dir='models', prefix='sentiment'):
    """
    Save trained model and vectorizer to disk
    
    Args:
        model: Trained model
        vectorizer: Text vectorizer
        model_dir (str): Directory to save model
        prefix (str): Prefix for model filename
    
    Returns:
        tuple: (model_path, vectorizer_path)
    """
    os.makedirs(model_dir, exist_ok=True)
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_path = os.path.join(model_dir, f"{prefix}_model_{timestamp}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save vectorizer
    vectorizer_path = os.path.join(model_dir, f"{prefix}_vectorizer_{timestamp}.pkl")
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")
    
    return model_path, vectorizer_path

def main():
    """
    Main function to train and evaluate sentiment analysis models
    """
    # Load and preprocess data
    train_df, val_df = load_and_preprocess_data(
        'data/twitter_training.csv',
        'data/twitter_validation.csv',
        remove_stopwords=True,
        lemmatization=True
    )
    
    # Split data if no validation set is provided
    if val_df is None:
        train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['sentiment'])
    
    # Get text and labels
    X_train = train_df['processed_tweet']
    y_train = train_df['sentiment']
    X_val = val_df['processed_tweet']
    y_val = val_df['sentiment']
    
    # Extract unique labels
    labels = sorted(train_df['sentiment'].unique())
    print(f"Labels: {labels}")
    
    # Vectorize text
    print("Vectorizing text data...")
    X_train_vec, X_val_vec, vectorizer = vectorize_text(
        X_train, X_val,
        method='tfidf',
        max_features=5000,
        min_df=5,
        ngram_range=(1, 2)
    )
    
    # Train model
    print("Training sentiment analysis model...")
    model = train_model(
        X_train_vec, y_train,
        model_type='lr',
        C=1.0,
        max_iter=1000,
        random_state=42
    )
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, X_val_vec, y_val, labels=labels)
    
    # Save model and vectorizer
    save_model(model, vectorizer)
    
    print("\nTraining complete!")
    
if __name__ == "__main__":
    main() 