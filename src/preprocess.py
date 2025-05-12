"""
Twitter Sentiment Analysis - Text Preprocessing
"""

import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download NLTK resources (uncomment first time)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

def preprocess_text(text, remove_stopwords=True, stemming=False, lemmatization=False):
    """
    Preprocess text by performing various cleaning operations
    
    Args:
        text (str): Input text to preprocess
        remove_stopwords (bool): Whether to remove stopwords
        stemming (bool): Whether to apply stemming
        lemmatization (bool): Whether to apply lemmatization
    
    Returns:
        str: Preprocessed text
    """
    # Handle NaN values
    if pd.isna(text):
        return ""
        
    # Convert to string if not already
    if not isinstance(text, str):
        text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    
    # Apply stemming
    if stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
    
    # Apply lemmatization
    if lemmatization:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back into a single string
    return ' '.join(tokens)

def load_and_preprocess_data(train_path, val_path=None, preprocess=True, **preprocess_kwargs):
    """
    Load and preprocess the Twitter sentiment analysis dataset
    
    Args:
        train_path (str): Path to training data
        val_path (str): Path to validation data
        preprocess (bool): Whether to preprocess the tweets
        **preprocess_kwargs: Additional arguments for preprocess_text function
    
    Returns:
        tuple: (train_df, val_df) - Processed DataFrames
    """
    # Load the datasets
    train_df = pd.read_csv(train_path, names=['id', 'topic', 'sentiment', 'tweet'], header=None)
    
    if val_path:
        val_df = pd.read_csv(val_path, names=['id', 'topic', 'sentiment', 'tweet'], header=None)
    else:
        val_df = None
    
    # Preprocess the tweets
    if preprocess:
        train_df['processed_tweet'] = train_df['tweet'].apply(
            lambda x: preprocess_text(x, **preprocess_kwargs)
        )
        
        if val_df is not None:
            val_df['processed_tweet'] = val_df['tweet'].apply(
                lambda x: preprocess_text(x, **preprocess_kwargs)
            )
    
    return train_df, val_df

if __name__ == "__main__":
    # Example usage
    train_df, val_df = load_and_preprocess_data(
        'data/twitter_training.csv',
        'data/twitter_validation.csv',
        remove_stopwords=True,
        stemming=False,
        lemmatization=True
    )
    
    print(f"Training data shape: {train_df.shape}")
    if val_df is not None:
        print(f"Validation data shape: {val_df.shape}")
    
    # Display sample preprocessing results
    print("\nSample preprocessing results:")
    for i, (original, processed) in enumerate(
        zip(train_df['tweet'].head(3), train_df['processed_tweet'].head(3))
    ):
        print(f"\nOriginal {i+1}: {original}")
        print(f"Processed {i+1}: {processed}") 