"""
Twitter Sentiment Analysis - Data Visualization
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Import local modules
from preprocess import load_and_preprocess_data

def create_wordclouds_by_sentiment(df, save_dir='visualizations'):
    """
    Create word clouds for each sentiment category
    
    Args:
        df (DataFrame): DataFrame containing tweets and sentiments
        save_dir (str): Directory to save wordcloud images
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get unique sentiment values
    sentiments = df['sentiment'].unique()
    
    plt.figure(figsize=(15, len(sentiments) * 5))
    
    for i, sentiment in enumerate(sentiments):
        # Filter tweets by sentiment
        sentiment_tweets = df[df['sentiment'] == sentiment]['processed_tweet']
        
        # Combine all tweets for this sentiment into a single string
        text = ' '.join(sentiment_tweets.tolist())
        
        # Generate and plot wordcloud
        plt.subplot(len(sentiments), 1, i+1)
        wordcloud = WordCloud(width=800, height=400, 
                              background_color='white',
                              max_words=100, 
                              contour_width=3, 
                              contour_color='steelblue').generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Word Cloud - {sentiment}', fontsize=16)
        plt.axis('off')
        
        # Save individual wordcloud
        wordcloud.to_file(os.path.join(save_dir, f'wordcloud_{sentiment}.png'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'all_wordclouds.png'))
    plt.close()

def plot_tweet_length_distribution(df, save_dir='visualizations'):
    """
    Plot distribution of tweet lengths by sentiment
    
    Args:
        df (DataFrame): DataFrame containing tweets and sentiments
        save_dir (str): Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate tweet lengths
    df['tweet_length'] = df['tweet'].apply(lambda x: len(str(x)))
    df['processed_tweet_length'] = df['processed_tweet'].apply(lambda x: len(str(x)))
    
    # Plot distribution of original tweet lengths
    plt.figure(figsize=(12, 8))
    sns.histplot(data=df, x='tweet_length', hue='sentiment', bins=50, kde=True)
    plt.title('Distribution of Tweet Lengths by Sentiment (Original)', fontsize=16)
    plt.xlabel('Tweet Length (characters)')
    plt.ylabel('Count')
    plt.xlim(0, df['tweet_length'].quantile(0.99))  # Limit x-axis to 99th percentile
    plt.savefig(os.path.join(save_dir, 'tweet_length_dist.png'))
    plt.close()
    
    # Plot distribution of processed tweet lengths
    plt.figure(figsize=(12, 8))
    sns.histplot(data=df, x='processed_tweet_length', hue='sentiment', bins=50, kde=True)
    plt.title('Distribution of Tweet Lengths by Sentiment (Processed)', fontsize=16)
    plt.xlabel('Processed Tweet Length (characters)')
    plt.ylabel('Count')
    plt.xlim(0, df['processed_tweet_length'].quantile(0.99))  # Limit x-axis to 99th percentile
    plt.savefig(os.path.join(save_dir, 'processed_tweet_length_dist.png'))
    plt.close()
    
    # Plot boxplot of tweet lengths by sentiment
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='sentiment', y='tweet_length')
    plt.title('Tweet Length by Sentiment (Original)', fontsize=16)
    plt.xlabel('Sentiment')
    plt.ylabel('Tweet Length (characters)')
    plt.savefig(os.path.join(save_dir, 'tweet_length_boxplot.png'))
    plt.close()
    
    # Plot boxplot of processed tweet lengths by sentiment
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='sentiment', y='processed_tweet_length')
    plt.title('Tweet Length by Sentiment (Processed)', fontsize=16)
    plt.xlabel('Sentiment')
    plt.ylabel('Processed Tweet Length (characters)')
    plt.savefig(os.path.join(save_dir, 'processed_tweet_length_boxplot.png'))
    plt.close()

def plot_top_words_by_sentiment(df, save_dir='visualizations', top_n=20):
    """
    Plot top words for each sentiment category
    
    Args:
        df (DataFrame): DataFrame containing tweets and sentiments
        save_dir (str): Directory to save visualizations
        top_n (int): Number of top words to plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get unique sentiments
    sentiments = df['sentiment'].unique()
    
    # Initialize CountVectorizer
    vectorizer = CountVectorizer(max_features=1000)
    
    # For each sentiment, find top words
    for sentiment in sentiments:
        # Filter tweets by sentiment
        sentiment_tweets = df[df['sentiment'] == sentiment]['processed_tweet']
        
        # Skip if no tweets
        if len(sentiment_tweets) == 0:
            continue
        
        # Vectorize the tweets
        X = vectorizer.fit_transform(sentiment_tweets)
        
        # Get feature names
        words = vectorizer.get_feature_names_out()
        
        # Sum up word counts across all documents
        total_counts = np.sum(X.toarray(), axis=0)
        
        # Create dictionary of word counts
        word_counts = dict(zip(words, total_counts))
        
        # Sort by count and get top N
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_words_df = pd.DataFrame(top_words, columns=['word', 'count'])
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(data=top_words_df, y='word', x='count', palette='viridis')
        plt.title(f'Top {top_n} Words - {sentiment}', fontsize=16)
        plt.xlabel('Count')
        plt.ylabel('Word')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'top_words_{sentiment}.png'))
        plt.close()

def plot_sentiment_distribution(df, save_dir='visualizations'):
    """
    Plot distribution of sentiments in the dataset
    
    Args:
        df (DataFrame): DataFrame containing tweets and sentiments
        save_dir (str): Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Count sentiments
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']
    
    # Plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=sentiment_counts, x='sentiment', y='count', palette='viridis')
    
    # Add count labels on top of bars
    for i, count in enumerate(sentiment_counts['count']):
        ax.text(i, count + 10, str(count), ha='center')
    
    plt.title('Distribution of Sentiments', fontsize=16)
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sentiment_distribution.png'))
    plt.close()
    
    # Also create a pie chart
    plt.figure(figsize=(10, 8))
    plt.pie(sentiment_counts['count'], labels=sentiment_counts['sentiment'], 
            autopct='%1.1f%%', startangle=90, shadow=True, 
            wedgeprops={'edgecolor': 'black'})
    plt.title('Sentiment Distribution (Pie Chart)', fontsize=16)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sentiment_distribution_pie.png'))
    plt.close()

def plot_dimensionality_reduction(df, save_dir='visualizations'):
    """
    Plot dimensionality reduction visualizations (t-SNE, PCA)
    
    Args:
        df (DataFrame): DataFrame containing tweets and sentiments
        save_dir (str): Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Sample tweets to make visualization faster (max 5000)
    if len(df) > 5000:
        sample_df = df.sample(5000, random_state=42)
    else:
        sample_df = df
    
    # Vectorize the tweets
    vectorizer = CountVectorizer(max_features=1000)
    X = vectorizer.fit_transform(sample_df['processed_tweet'])
    
    # Get sentiment labels
    y = sample_df['sentiment']
    
    # 1. PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X.toarray())
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis', alpha=0.6)
    plt.title('PCA Visualization of Tweets', fontsize=16)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Sentiment')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pca_visualization.png'))
    plt.close()
    
    # 2. t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X.toarray())
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='viridis', alpha=0.6)
    plt.title('t-SNE Visualization of Tweets', fontsize=16)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title='Sentiment')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tsne_visualization.png'))
    plt.close()
    
    # 3. TruncatedSVD (similar to PCA but works with sparse matrices)
    svd = TruncatedSVD(n_components=2, random_state=42)
    X_svd = svd.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_svd[:, 0], y=X_svd[:, 1], hue=y, palette='viridis', alpha=0.6)
    plt.title('SVD Visualization of Tweets', fontsize=16)
    plt.xlabel('SVD Component 1')
    plt.ylabel('SVD Component 2')
    plt.legend(title='Sentiment')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'svd_visualization.png'))
    plt.close()

def plot_topic_distribution(df, save_dir='visualizations'):
    """
    Plot distribution of topics and sentiment by topic
    
    Args:
        df (DataFrame): DataFrame containing tweets and topics
        save_dir (str): Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Count topics
    topic_counts = df['topic'].value_counts().reset_index()
    topic_counts.columns = ['topic', 'count']
    
    # Limit to top 20 topics if there are many
    if len(topic_counts) > 20:
        topic_counts = topic_counts.head(20)
    
    # Plot topic distribution
    plt.figure(figsize=(12, 8))
    sns.barplot(data=topic_counts, x='count', y='topic')
    plt.title('Distribution of Topics (Top 20)', fontsize=16)
    plt.xlabel('Count')
    plt.ylabel('Topic')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'topic_distribution.png'))
    plt.close()
    
    # Plot sentiment distribution by topic (top 10 topics)
    if len(topic_counts) > 10:
        top_topics = topic_counts.head(10)['topic'].tolist()
    else:
        top_topics = topic_counts['topic'].tolist()
    
    # Filter only top topics
    filtered_df = df[df['topic'].isin(top_topics)]
    
    # Create count table
    topic_sentiment = pd.crosstab(filtered_df['topic'], filtered_df['sentiment'])
    
    # Convert to percentage
    topic_sentiment_pct = topic_sentiment.div(topic_sentiment.sum(axis=1), axis=0) * 100
    
    # Plot percentage stacked bar chart
    plt.figure(figsize=(12, 8))
    topic_sentiment_pct.plot(kind='barh', stacked=True, colormap='viridis')
    plt.title('Sentiment Distribution by Topic (Top 10)', fontsize=16)
    plt.xlabel('Percentage')
    plt.ylabel('Topic')
    plt.legend(title='Sentiment')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sentiment_by_topic_percent.png'))
    plt.close()
    
    # Plot count stacked bar chart
    plt.figure(figsize=(12, 8))
    topic_sentiment.plot(kind='barh', stacked=True, colormap='viridis')
    plt.title('Sentiment Count by Topic (Top 10)', fontsize=16)
    plt.xlabel('Count')
    plt.ylabel('Topic')
    plt.legend(title='Sentiment')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sentiment_by_topic_count.png'))
    plt.close()

def main():
    """
    Main function to generate all visualizations
    """
    # Create visualization directory
    os.makedirs('visualizations', exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_df, val_df = load_and_preprocess_data(
        'data/twitter_training.csv',
        'data/twitter_validation.csv',
        remove_stopwords=True,
        lemmatization=True
    )
    
    # Combine datasets for visualization
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    
    # Generate visualizations
    print("\nGenerating wordclouds by sentiment...")
    create_wordclouds_by_sentiment(combined_df)
    
    print("Plotting tweet length distributions...")
    plot_tweet_length_distribution(combined_df)
    
    print("Plotting top words by sentiment...")
    plot_top_words_by_sentiment(combined_df)
    
    print("Plotting sentiment distribution...")
    plot_sentiment_distribution(combined_df)
    
    print("Plotting topic distributions...")
    plot_topic_distribution(combined_df)
    
    print("Generating dimensionality reduction visualizations...")
    plot_dimensionality_reduction(combined_df)
    
    print("\nAll visualizations generated and saved to 'visualizations/' directory!")

if __name__ == "__main__":
    main() 