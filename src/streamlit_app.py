"""
Twitter Sentiment Analysis - Streamlit Web App
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import time
from src.optimized_model import load_optimized_model
from src.preprocess import preprocess_text, load_and_preprocess_data

# Set page config
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define color scheme
COLORS = {
    "Positive": "#28a745",  # Green
    "Negative": "#dc3545",  # Red
    "Neutral": "#17a2b8",   # Blue
    "Irrelevant": "#6c757d" # Gray
}

# Apply custom CSS
st.markdown("""
<style>
.sentiment-positive {
    color: #28a745;
    font-weight: bold;
}
.sentiment-negative {
    color: #dc3545;
    font-weight: bold;
}
.sentiment-neutral {
    color: #17a2b8;
    font-weight: bold;
}
.sentiment-irrelevant {
    color: #6c757d;
    font-weight: bold;
}
.confidence-high {
    background-color: rgba(40, 167, 69, 0.2);
    padding: 2px 5px;
    border-radius: 3px;
}
.confidence-medium {
    background-color: rgba(255, 193, 7, 0.2);
    padding: 2px 5px;
    border-radius: 3px;
}
.confidence-low {
    background-color: rgba(220, 53, 69, 0.2);
    padding: 2px 5px;
    border-radius: 3px;
}
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    margin-bottom: 1rem;
}
.subheader {
    font-size: 1.8rem;
    font-weight: bold;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}
.result-card {
    padding: 1.5rem;
    border-radius: 0.5rem;
    background-color: #f8f9fa;
    box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Function to format sentiment with color
def format_sentiment(sentiment):
    sentiment_lower = sentiment.lower()
    return f'<span class="sentiment-{sentiment_lower}">{sentiment}</span>'

# Function to format confidence level
def format_confidence(confidence):
    if confidence >= 0.7:
        return f'<span class="confidence-high">{confidence:.2f}</span>'
    elif confidence >= 0.4:
        return f'<span class="confidence-medium">{confidence:.2f}</span>'
    else:
        return f'<span class="confidence-low">{confidence:.2f}</span>'

# Function to predict sentiment
def predict_sentiment(text, model):
    # Make prediction
    prediction = model.predict([text])[0]
    
    # Get probability if available
    probability = None
    if hasattr(model['classifier'], 'predict_proba'):
        probabilities = model.predict_proba([text])[0]
        probability = max(probabilities)
    
    return prediction, probability

# Function to predict batch of texts
def predict_batch(texts, model):
    # Make predictions
    predictions = model.predict(texts)
    
    # Get probabilities if available
    probabilities = None
    if hasattr(model['classifier'], 'predict_proba'):
        proba_results = model.predict_proba(texts)
        probabilities = [max(proba) for proba in proba_results]
    
    return predictions, probabilities

# Function to load model (with caching)
@st.cache_resource
def load_model(model_path='models/optimized_model.pkl'):
    if not os.path.exists(model_path):
        return None
    return load_optimized_model(model_path)

# Function to load feature importance data
@st.cache_data
def load_feature_importance():
    try:
        return pd.read_csv('analysis/feature_importance.csv')
    except FileNotFoundError:
        return None

# Function to generate word cloud from texts
def generate_wordcloud(texts, sentiment=None, max_words=100):
    if sentiment:
        title = f"Word Cloud - {sentiment} Sentiment"
    else:
        title = "Word Cloud - All Texts"
    
    # Join all texts
    text = ' '.join(texts)
    
    # Generate wordcloud
    color = COLORS.get(sentiment, "viridis")
    wc = WordCloud(
        max_words=max_words,
        background_color='white',
        width=800,
        height=400,
        colormap=color if sentiment is None else None,
        color_func=lambda *args, **kwargs: color if sentiment else None,
        contour_width=1,
        contour_color='steelblue'
    ).generate(text)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.set_title(title)
    ax.axis('off')
    
    return fig

# Function to display model info
def display_model_info():
    st.markdown("<div class='subheader'>Model Information</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Model Type**: Random Forest  
        **Vectorizer**: N-gram Bag of Words (1-3 grams)  
        **Accuracy**: 92.6%  
        """)
    
    with col2:
        st.info("""
        **Classes**: Positive, Negative, Neutral, Irrelevant  
        **Features**: 10,000 n-grams   
        **F1-Score**: 0.93 (weighted avg)  
        """)

# Function to display feature importance
def display_feature_importance():
    importance_df = load_feature_importance()
    
    if importance_df is not None:
        st.markdown("<div class='subheader'>Top Features</div>", unsafe_allow_html=True)
        
        # Feature importance chart
        fig, ax = plt.subplots(figsize=(10, 6))
        chart_data = importance_df.head(15)  # Top 15 features
        sns.barplot(data=chart_data, y='feature', x='importance', ax=ax, palette='viridis')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        ax.set_title('Top 15 Important Features')
        st.pyplot(fig)
        
        # Word cloud of top features
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        # Create dictionary of feature: importance
        feature_importance = dict(zip(importance_df['feature'], importance_df['importance']))
        wc = WordCloud(
            max_words=50,
            background_color='white',
            width=800,
            height=400,
            colormap='viridis',
            contour_width=1,
            contour_color='steelblue'
        ).generate_from_frequencies(feature_importance)
        ax2.imshow(wc, interpolation='bilinear')
        ax2.set_title('Word Cloud of Important Features')
        ax2.axis('off')
        st.pyplot(fig2)
    else:
        st.warning("Feature importance data not available. Run optimized_model.py first.")

# Function to display confusion matrix (cached)
@st.cache_data
def load_confusion_matrix():
    try:
        conf_matrix = pd.read_csv('analysis/confusion_matrix.csv', index_col=0)
        return conf_matrix
    except FileNotFoundError:
        try:
            # Try to find the png and display that instead
            if os.path.exists('analysis/confusion_matrix.png'):
                return 'analysis/confusion_matrix.png'
        except:
            pass
        return None

def display_confusion_matrix():
    conf_matrix = load_confusion_matrix()
    
    if conf_matrix is not None:
        st.markdown("<div class='subheader'>Confusion Matrix</div>", unsafe_allow_html=True)
        
        if isinstance(conf_matrix, str):
            # It's an image path
            st.image(conf_matrix)
        else:
            # It's a DataFrame
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
    else:
        st.warning("Confusion matrix not available. Run optimized_model.py first.")

# Function to generate sample tweets for demonstration
def generate_sample_tweets():
    return [
        "I absolutely love this new feature! It's amazing!",
        "This product is terrible. The worst purchase I've ever made.",
        "The service was okay. Nothing special but it gets the job done.",
        "I'm not sure how I feel about this update. It has some good points and some bad.",
        "Just watched the latest movie in the series and it was fantastic! Best one yet!",
        "The customer service at this company is non-existent. No one responds to emails.",
        "It works as advertised. Simple and straightforward.",
        "Why would anyone think this is a good idea? Complete disaster!",
        "The concert last night was absolutely mind-blowing! I'm still in awe.",
        "The website is down again. This happens every single week. So frustrating.",
    ]

# Function to display error analysis
def display_error_analysis():
    # Try to load error rates by class
    try:
        # Check if error_rate_by_class.png exists
        if os.path.exists('analysis/error_rate_by_class.png'):
            st.markdown("<div class='subheader'>Error Analysis</div>", unsafe_allow_html=True)
            st.image('analysis/error_rate_by_class.png')
            return True
    except:
        pass
    
    # If no error analysis is available, try to read misclassified examples
    try:
        errors_df = pd.read_csv('analysis/misclassified_examples.csv')
        if not errors_df.empty:
            st.markdown("<div class='subheader'>Misclassification Examples</div>", unsafe_allow_html=True)
            st.dataframe(errors_df.head(10))
            return True
    except:
        pass
    
    return False

# Main app
def main():
    # Title and description
    st.markdown("<div class='main-header'>Twitter Sentiment Analysis</div>", unsafe_allow_html=True)
    st.markdown(
        "An intelligent tool for analyzing sentiment in tweets using advanced machine learning. "
        "Find out if text conveys positive, negative, neutral, or irrelevant sentiment."
    )
    
    # Sidebar
    with st.sidebar:
        # Use local Twitter icon base64 string instead of URL
        st.markdown("""
        <img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNDggMjA0Ij48cGF0aCBkPSJNMjIxLjk1IDUxLjI5Yy4xNSAyLjE3LjE1IDQuMzQuMTUgNi41MyAwIDY2LjczLTUwLjggMTQzLjY5LTE0My42OSAxNDMuNjl2LS4wNGMtMjcuNDQuMDQtNTQuMzEtNy44Mi03Ny40MS0yMi42NCA0LjA0LjQ4IDguMDkuNzIgMTIuMTUuNzIgMjMuNDcgMCA0Ni4xOC04LjAxIDY0LjU1LTIyLjY0LTIyLjA2LS40MS00MS42OS0xNC44LTQ4LjM4LTM1LjA3IDcuNjggMS40NiAxNS41NSAxLjE2IDIzLjA2LS44OS0yNC4xMi00Ljg3LTQxLjIzLTI2LjAzLTQxLjIzLTQ5Ljgydi0uNjNjNy4xIDMuOTYgMTUuMDcgNi4xNCAyMy40MiA2LjM0LTIzLjEzLTE1LjQ2LTMwLjI2LTQ1Ljk4LTE2LjE0LTY5Ljc3IDI1LjY0IDMxLjU1IDYzLjQ3IDUwLjczIDEwNC4wOCA1Mi43Ni00LjA3LTE3LjU0IDEuNDktMzUuOTIgMTQuNjEtNDguMjUgMjAuMzQtMTkuMTIgNTIuMzMtMTguMTQgNzEuNDUgMi4xOSAxMS4zMS0yLjIzIDIyLjE1LTYuMzggMzIuMDctMTIuMjYtMy43NyAxMS42OS0xMS42NiAyMS42Mi0yMi4yIDI3LjkzIDEwLjAxLTEuMTggMTkuNzktMy44NiAyOS03Ljk1LTYuNyAxMC4xNi0xNS4xIDE5LjAyLTI0LjY4IDI2LjA5eiIgZmlsbD0iIzFkOWJmMCIvPjwvc3ZnPg==" width="50" height="50">
        """, unsafe_allow_html=True)
        st.title("Navigation")
        
        # Navigation
        page = st.radio(
            "Select Page",
            ["💬 Text Analysis", "📊 Batch Analysis", "ℹ️ Model Information"]
        )
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Model not found. Please run `python src/optimized_model.py` to train the model first.")
        st.info("You can still explore the app's UI, but predictions will not work without a trained model.")
        # Return here to prevent errors if no model is found
        return
    
    # Display different pages based on selection
    if page == "💬 Text Analysis":
        st.markdown("<div class='subheader'>Text Analysis</div>", unsafe_allow_html=True)
        
        # Text input
        user_input = st.text_area("Enter text to analyze", 
                                  height=100, 
                                  placeholder="Type or paste text here...")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            analyze_button = st.button("Analyze", type="primary", use_container_width=True)
        with col2:
            example_button = st.button("Try Example", use_container_width=False)
        
        if example_button:
            user_input = "I just tried the new updates, and they're absolutely fantastic! The interface is so much cleaner and easier to use. Loving it so far!"
            st.session_state.example_used = True
            st.experimental_rerun()
        
        if 'example_used' in st.session_state and st.session_state.example_used:
            st.session_state.example_used = False
            user_input = "I just tried the new updates, and they're absolutely fantastic! The interface is so much cleaner and easier to use. Loving it so far!"
            analyze_button = True
            
        if analyze_button and user_input:
            with st.spinner("Analyzing..."):
                # Make prediction
                sentiment, confidence = predict_sentiment(user_input, model)
                
                # Preprocess text (for display)
                preprocessed = preprocess_text(user_input)
                
                # Display result in card format
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                
                # Results header
                st.markdown("<div class='subheader'>Analysis Result</div>", unsafe_allow_html=True)
                
                # Two columns for results
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    st.markdown("**Sentiment:**")
                    st.markdown(format_sentiment(sentiment), unsafe_allow_html=True)
                
                with result_col2:
                    st.markdown("**Confidence:**")
                    if confidence:
                        st.markdown(format_confidence(confidence), unsafe_allow_html=True)
                    else:
                        st.text("N/A")
                
                # Original text
                st.markdown("**Original Text:**")
                st.markdown(f"> {user_input}")
                
                # Preprocessed text (collapsible)
                with st.expander("Show Preprocessed Text"):
                    st.text(preprocessed)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Visualize words in the input text
                if preprocessed:
                    st.markdown("<div class='subheader'>Word Cloud</div>", unsafe_allow_html=True)
                    word_cloud_fig = generate_wordcloud([preprocessed], sentiment)
                    st.pyplot(word_cloud_fig)
                    
                # Explain the prediction (simplified)
                st.markdown("<div class='subheader'>Explanation</div>", unsafe_allow_html=True)
                
                # Get feature importance
                importance_df = load_feature_importance()
                if importance_df is not None:
                    # Extract words from the text that are in the top features
                    top_features = set(importance_df['feature'].head(100))
                    words_in_text = preprocessed.split()
                    features_found = [word for word in words_in_text if word in top_features]
                    
                    if features_found:
                        st.markdown("The prediction was influenced by these important words in your text:")
                        for word in features_found[:5]:  # Show top 5 matching features
                            feature_importance = importance_df[importance_df['feature'] == word]['importance'].values
                            if len(feature_importance) > 0:
                                st.markdown(f"- **{word}** (importance: {feature_importance[0]:.4f})")
                    else:
                        st.markdown("No specific high-importance words were found in your text. The prediction is based on overall patterns in the language.")
                else:
                    st.markdown("Predictions are based on language patterns learned from thousands of tweets.")
    
    elif page == "📊 Batch Analysis":
        st.markdown("<div class='subheader'>Batch Analysis</div>", unsafe_allow_html=True)
        
        upload_col, example_col = st.columns([3, 1])
        
        with upload_col:
            # File upload option
            uploaded_file = st.file_uploader("Upload CSV file with a 'text' column", type=["csv"])
        
        with example_col:
            # Example button
            if st.button("Use Example Data", use_container_width=True):
                # Load example data
                try:
                    example_df = pd.read_csv("sample_tweets.csv")
                    st.session_state.batch_data = example_df
                    st.success("Example data loaded!")
                except FileNotFoundError:
                    st.error("Example file 'sample_tweets.csv' not found.")
        
        # Process uploaded file
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if 'text' not in df.columns:
                    st.error("CSV file must contain a 'text' column.")
                else:
                    st.session_state.batch_data = df
                    st.success("File uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading file: {e}")
        
        # Process batch data if available
        if 'batch_data' in st.session_state:
            df = st.session_state.batch_data
            
            # Preview data
            st.markdown("<div class='subheader'>Data Preview</div>", unsafe_allow_html=True)
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("Analyze Batch", type="primary"):
                with st.spinner("Analyzing batch data..."):
                    # Get texts
                    texts = df['text'].tolist()
                    
                    # Make predictions
                    predictions, probabilities = predict_batch(texts, model)
                    
                    # Add predictions to dataframe
                    results_df = df.copy()
                    results_df['sentiment'] = predictions
                    if probabilities:
                        results_df['confidence'] = probabilities
                    
                    # Store in session state for persistence
                    st.session_state.batch_results = results_df
            
            # If results exist, display them
            if 'batch_results' in st.session_state:
                results_df = st.session_state.batch_results
                
                # Results
                st.markdown("<div class='subheader'>Analysis Results</div>", unsafe_allow_html=True)
                
                # Results table with styling
                st.dataframe(
                    results_df.style.format({'confidence': '{:.2f}'})
                                 .background_gradient(cmap='Greens', subset=['confidence']),
                    use_container_width=True
                )
                
                # Download option
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results CSV",
                    data=csv,
                    file_name="sentiment_analysis_results.csv",
                    mime="text/csv"
                )
                
                # Visualizations
                st.markdown("<div class='subheader'>Visualizations</div>", unsafe_allow_html=True)
                
                # Create tabs for different visualizations
                viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Distribution", "Word Clouds", "Confidence Analysis"])
                
                with viz_tab1:
                    # Count sentiment distribution
                    sentiment_counts = results_df['sentiment'].value_counts()
                    
                    # Create two columns for charts
                    dist_col1, dist_col2 = st.columns(2)
                    
                    with dist_col1:
                        # Bar chart
                        fig, ax = plt.subplots(figsize=(7, 5))
                        ax = sentiment_counts.plot(
                            kind='bar',
                            color=[COLORS.get(s, "#6c757d") for s in sentiment_counts.index],
                            ax=ax
                        )
                        ax.set_xlabel('Sentiment')
                        ax.set_ylabel('Count')
                        ax.set_title('Sentiment Distribution')
                        
                        # Add count labels
                        for i, count in enumerate(sentiment_counts):
                            ax.text(i, count + 0.1, str(count), ha='center', va='bottom')
                        
                        st.pyplot(fig)
                    
                    with dist_col2:
                        # Pie chart
                        fig, ax = plt.subplots(figsize=(7, 5))
                        sentiment_counts.plot(
                            kind='pie',
                            autopct='%1.1f%%',
                            colors=[COLORS.get(s, "#6c757d") for s in sentiment_counts.index],
                            ax=ax
                        )
                        ax.set_title('Sentiment Distribution')
                        ax.set_ylabel('')  # Hide "None" label
                        st.pyplot(fig)
                
                with viz_tab2:
                    # Generate word clouds for each sentiment
                    sentiments = sorted(results_df['sentiment'].unique())
                    
                    for sentiment in sentiments:
                        sentiment_texts = results_df[results_df['sentiment'] == sentiment]['text'].apply(preprocess_text).tolist()
                        if sentiment_texts:
                            fig = generate_wordcloud(sentiment_texts, sentiment)
                            st.pyplot(fig)
                
                with viz_tab3:
                    if 'confidence' in results_df.columns:
                        # Create violin plot of confidence by sentiment
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.violinplot(
                            data=results_df, 
                            x='sentiment', 
                            y='confidence',
                            palette=[COLORS.get(s, "#6c757d") for s in sentiments],
                            ax=ax
                        )
                        ax.set_title('Confidence Distribution by Sentiment')
                        ax.set_ylim(0, 1)
                        st.pyplot(fig)
                        
                        # Confidence thresholds
                        st.markdown("### Confidence Analysis")
                        
                        # Count by confidence levels
                        low_conf = len(results_df[results_df['confidence'] < 0.4])
                        med_conf = len(results_df[(results_df['confidence'] >= 0.4) & (results_df['confidence'] < 0.7)])
                        high_conf = len(results_df[results_df['confidence'] >= 0.7])
                        
                        conf_data = pd.DataFrame({
                            'Confidence Level': ['Low (<0.4)', 'Medium (0.4-0.7)', 'High (>0.7)'],
                            'Count': [low_conf, med_conf, high_conf]
                        })
                        
                        # Bar chart of confidence levels
                        fig, ax = plt.subplots(figsize=(8, 5))
                        sns.barplot(
                            data=conf_data,
                            x='Confidence Level',
                            y='Count',
                            palette=['#dc3545', '#ffc107', '#28a745'],
                            ax=ax
                        )
                        ax.set_title('Prediction Confidence Levels')
                        
                        # Add percentage labels
                        total = len(results_df)
                        for i, count in enumerate([low_conf, med_conf, high_conf]):
                            percentage = count / total * 100
                            ax.text(i, count + 0.1, f"{percentage:.1f}%", ha='center', va='bottom')
                        
                        st.pyplot(fig)
    
    elif page == "ℹ️ Model Information":
        st.markdown("<div class='subheader'>Model Details</div>", unsafe_allow_html=True)
        
        # Create tabs for different aspects of the model
        info_tab1, info_tab2, info_tab3 = st.tabs(["Overview", "Performance", "Features"])
        
        with info_tab1:
            display_model_info()
            
            # Model explanation
            st.markdown("### How It Works")
            st.markdown(
                """
                This sentiment analysis model uses a **Random Forest classifier** with **N-gram features** to identify the emotional tone of text. 
                
                The system follows these steps:
                1. **Preprocessing**: Clean and normalize text (remove URLs, hashtags, etc.)
                2. **Feature Extraction**: Convert text to n-gram features (1-3 word combinations)
                3. **Classification**: Predict sentiment using the trained Random Forest model
                4. **Confidence Scoring**: Estimate prediction reliability with probability scores
                
                The model was trained on a dataset of over **70,000 labeled tweets** covering various topics and sentiments.
                """
            )
        
        with info_tab2:
            # Performance metrics
            st.markdown("### Performance Metrics")
            
            # Create metrics in three columns
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric(label="Overall Accuracy", value="92.6%")
                
            with metric_col2:
                st.metric(label="F1 Score (weighted)", value="0.93")
                
            with metric_col3:
                st.metric(label="Precision (weighted)", value="0.93")
            
            # Class-specific metrics
            metrics_col1, metrics_col2 = st.columns(2)
            
            with metrics_col1:
                st.markdown("**Precision by Class:**")
                st.markdown("- Positive: 0.89")
                st.markdown("- Negative: 0.89")
                st.markdown("- Neutral: 0.97")
                st.markdown("- Irrelevant: 0.99")
            
            with metrics_col2:
                st.markdown("**Recall by Class:**")
                st.markdown("- Positive: 0.97")
                st.markdown("- Negative: 0.95")
                st.markdown("- Neutral: 0.90")
                st.markdown("- Irrelevant: 0.87")
            
            # Confusion Matrix
            display_confusion_matrix()
            
            # Error Analysis
            if not display_error_analysis():
                st.markdown("### Error Analysis")
                st.markdown(
                    """
                    The model performs best on **Positive** sentiment (96.8% accuracy) and worst on **Irrelevant** sentiment (86.6% accuracy).
                    
                    Common misclassification patterns:
                    - Neutral tweets incorrectly classified as Negative
                    - Irrelevant tweets incorrectly classified as Positive
                    - Neutral tweets incorrectly classified as Positive
                    """
                )
        
        with info_tab3:
            display_feature_importance()
            
            # Explanation of features
            st.markdown("### Understanding Features")
            st.markdown(
                """
                The model primarily uses **word presence** and **word combinations** (n-grams) as features to classify sentiment.
                
                Key observations:
                - Emotive words like "love", "hate", "good", "bad" are strong predictors
                - Profanity is often associated with negative sentiment
                - Product-related terms ("fix", "server", "game") can indicate specific topics with sentiment patterns
                - Some neutral words can become important when combined with sentiment modifiers
                """
            )
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Twitter Sentiment Analysis**  
    Created with Streamlit and scikit-learn  
    Made with ❤️ for data science
    
    [View on GitHub](https://github.com/yourusername/twitter-sentiment-analysis)
    """)

if __name__ == "__main__":
    main() 