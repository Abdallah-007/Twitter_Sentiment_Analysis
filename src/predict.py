"""
Twitter Sentiment Analysis - Prediction Module
"""

import os
import pickle
import pandas as pd
import numpy as np
from preprocess import preprocess_text

class SentimentPredictor:
    """
    Class for making sentiment predictions using a trained model
    """
    
    def __init__(self, model_path, vectorizer_path):
        """
        Initialize the sentiment predictor
        
        Args:
            model_path (str): Path to trained model file
            vectorizer_path (str): Path to trained vectorizer file
        """
        # Load the model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load the vectorizer
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
    
    def predict_sentiment(self, text, preprocess=True, **preprocess_kwargs):
        """
        Predict sentiment for a single text input
        
        Args:
            text (str): Input text
            preprocess (bool): Whether to preprocess the text
            **preprocess_kwargs: Arguments for preprocess_text function
        
        Returns:
            str: Predicted sentiment label
        """
        # Preprocess the text if required
        if preprocess:
            text = preprocess_text(text, **preprocess_kwargs)
        
        # Vectorize the text
        text_vec = self.vectorizer.transform([text])
        
        # Make prediction
        sentiment = self.model.predict(text_vec)[0]
        
        # Get prediction probability if available
        probability = None
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(text_vec)[0]
            probability = probabilities.max()
        
        return sentiment, probability
    
    def predict_batch(self, texts, preprocess=True, **preprocess_kwargs):
        """
        Predict sentiment for a batch of text inputs
        
        Args:
            texts (list): List of input texts
            preprocess (bool): Whether to preprocess the texts
            **preprocess_kwargs: Arguments for preprocess_text function
        
        Returns:
            pandas.DataFrame: DataFrame with original texts and predictions
        """
        # Preprocess texts if required
        if preprocess:
            processed_texts = [preprocess_text(text, **preprocess_kwargs) for text in texts]
        else:
            processed_texts = texts
        
        # Vectorize the texts
        texts_vec = self.vectorizer.transform(processed_texts)
        
        # Make predictions
        sentiments = self.model.predict(texts_vec)
        
        # Get prediction probabilities if available
        probabilities = None
        if hasattr(self.model, 'predict_proba'):
            prob_values = self.model.predict_proba(texts_vec)
            probabilities = [prob.max() for prob in prob_values]
        
        # Create result DataFrame
        results = pd.DataFrame({
            'text': texts,
            'sentiment': sentiments
        })
        
        if probabilities is not None:
            results['probability'] = probabilities
        
        return results


def find_latest_model(model_dir='models', prefix='sentiment'):
    """
    Find the latest trained model and vectorizer
    
    Args:
        model_dir (str): Directory containing models
        prefix (str): Prefix for model filename
    
    Returns:
        tuple: (model_path, vectorizer_path)
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory {model_dir} not found")
    
    # Find all model files
    model_files = [f for f in os.listdir(model_dir) if f.startswith(f"{prefix}_model_")]
    
    if not model_files:
        raise FileNotFoundError("No model files found")
    
    # Get the latest model file
    latest_model = sorted(model_files)[-1]
    
    # Extract full timestamp from filename
    # Expected format: sentiment_model_YYYYMMDD_HHMMSS.pkl
    timestamp = latest_model.replace(f"{prefix}_model_", "").replace(".pkl", "")
    
    # Construct path to corresponding vectorizer
    vectorizer_file = f"{prefix}_vectorizer_{timestamp}.pkl"
    vectorizer_path = os.path.join(model_dir, vectorizer_file)
    
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Vectorizer file {vectorizer_path} not found")
    
    return os.path.join(model_dir, latest_model), vectorizer_path


def main():
    """
    Example usage of the sentiment predictor
    """
    try:
        # Find latest model and vectorizer
        model_path, vectorizer_path = find_latest_model()
        
        # Initialize the predictor
        predictor = SentimentPredictor(model_path, vectorizer_path)
        
        # Example texts for prediction
        example_texts = [
            "I absolutely love this new feature! It's amazing!",
            "This is the worst service I've ever experienced.",
            "The product is okay, nothing special but gets the job done.",
            "I'm not sure how I feel about this yet."
        ]
        
        # Make predictions
        print("Making predictions for example texts:")
        for text in example_texts:
            sentiment, probability = predictor.predict_sentiment(
                text,
                remove_stopwords=True,
                lemmatization=True
            )
            
            prob_str = f" (Probability: {probability:.4f})" if probability is not None else ""
            print(f"Text: {text}")
            print(f"Sentiment: {sentiment}{prob_str}\n")
        
        # Batch prediction example
        results = predictor.predict_batch(example_texts)
        print("\nBatch prediction results:")
        print(results)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to train a model first by running model.py")


if __name__ == "__main__":
    main() 