"""
Twitter Sentiment Analysis - Optimized Model Pipeline
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import local modules
from preprocess import load_and_preprocess_data

def create_optimized_pipeline():
    """
    Create an optimized pipeline based on the best parameters from tuning
    
    Returns:
        sklearn.pipeline.Pipeline: Optimized pipeline
    """
    # Create pipeline with the best model and vectorizer
    # Based on our tuning, the best was Random Forest with N-gram BoW
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(max_features=10000, ngram_range=(1, 3))),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=10,
            min_samples_leaf=2,
            random_state=42
        ))
    ])
    
    return pipeline

def train_and_evaluate(pipeline, X_train, y_train, X_val, y_val):
    """
    Train and evaluate the optimized pipeline
    
    Args:
        pipeline: Optimized sklearn pipeline
        X_train: Training text data
        y_train: Training labels
        X_val: Validation text data
        y_val: Validation labels
    
    Returns:
        tuple: (pipeline, y_pred, accuracy, report)
    """
    # Train the pipeline
    print("Training optimized model...")
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_val)
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, output_dict=True)
    
    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    return pipeline, y_pred, accuracy, report

def analyze_errors(X_val, y_val, y_pred, save_dir='analysis'):
    """
    Analyze misclassified examples
    
    Args:
        X_val: Validation text data
        y_val: True labels
        y_pred: Predicted labels
        save_dir: Directory to save analysis results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create DataFrame with validation data and predictions
    results_df = pd.DataFrame({
        'text': X_val,
        'true_label': y_val,
        'predicted_label': y_pred,
        'correct': y_val == y_pred
    })
    
    # Filter misclassified examples
    misclassified_df = results_df[~results_df['correct']]
    
    # Save misclassified examples to CSV
    misclassified_df.to_csv(os.path.join(save_dir, 'misclassified_examples.csv'), index=False)
    
    # Count misclassifications by class
    error_matrix = pd.crosstab(
        misclassified_df['true_label'], 
        misclassified_df['predicted_label'],
        rownames=['True'], 
        colnames=['Predicted']
    )
    
    # Save error matrix to CSV
    error_matrix.to_csv(os.path.join(save_dir, 'error_matrix.csv'))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    conf_mat = confusion_matrix(y_val, y_pred)
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(set(y_val)),
                yticklabels=sorted(set(y_val)))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    
    # Calculate error rates by class
    class_error_rates = {}
    for cls in sorted(set(y_val)):
        cls_indices = [i for i, label in enumerate(y_val) if label == cls]
        cls_true = [y_val[i] for i in cls_indices]
        cls_pred = [y_pred[i] for i in cls_indices]
        cls_accuracy = accuracy_score(cls_true, cls_pred)
        class_error_rates[cls] = 1 - cls_accuracy
    
    # Plot error rates by class
    plt.figure(figsize=(10, 6))
    classes = list(class_error_rates.keys())
    error_rates = list(class_error_rates.values())
    sns.barplot(x=classes, y=error_rates)
    plt.xlabel('Class')
    plt.ylabel('Error Rate')
    plt.title('Error Rate by Class')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_rate_by_class.png'))
    
    # Print analysis summary
    total_examples = len(y_val)
    misclassified_count = len(misclassified_df)
    print(f"\nError Analysis Summary:")
    print(f"Total validation examples: {total_examples}")
    print(f"Correctly classified: {total_examples - misclassified_count} ({(total_examples - misclassified_count) / total_examples:.2%})")
    print(f"Misclassified: {misclassified_count} ({misclassified_count / total_examples:.2%})")
    print("\nError rate by class:")
    for cls, error_rate in class_error_rates.items():
        print(f"  {cls}: {error_rate:.4f}")
    
    return misclassified_df, error_matrix

def save_optimized_model(pipeline, output_dir='models'):
    """
    Save the optimized model pipeline
    
    Args:
        pipeline: Trained pipeline
        output_dir: Directory to save the model
    
    Returns:
        str: Path to saved model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the pipeline
    model_path = os.path.join(output_dir, 'optimized_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    
    print(f"Optimized model saved to {model_path}")
    return model_path

def create_dummy_model():
    """
    Create a simple dummy model when the optimized model is not available
    Returns a pipeline that can be used as a fallback
    """
    # Create a simple model with minimal functionality
    vectorizer = CountVectorizer(max_features=100)
    classifier = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Create a pipeline
    model = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])
    
    # Train on a tiny dataset
    texts = [
        "I love this product, it's amazing!",
        "This is terrible, worst purchase ever.",
        "It's okay, nothing special.",
        "Not sure how I feel about this."
    ]
    labels = ["Positive", "Negative", "Neutral", "Irrelevant"]
    
    # Fit the model
    model.fit(texts, labels)
    
    return model

def load_optimized_model(model_path='models/optimized_model.pkl'):
    """
    Load the optimized model from a pickle file
    
    Args:
        model_path (str): Path to the model pickle file
    
    Returns:
        sklearn.pipeline.Pipeline: Trained model pipeline
    """
    try:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        else:
            print(f"Model file {model_path} not found. Creating a dummy model.")
            return create_dummy_model()
    except Exception as e:
        print(f"Error loading model: {e}. Creating a dummy model.")
        return create_dummy_model()

def analyze_feature_importance(pipeline, save_dir='analysis'):
    """
    Analyze feature importance from the model
    
    Args:
        pipeline: Trained pipeline
        save_dir: Directory to save analysis results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Check if the model has feature_importances_ attribute (RandomForest does)
    if hasattr(pipeline['classifier'], 'feature_importances_'):
        # Get feature names from vectorizer
        vectorizer = pipeline['vectorizer']
        feature_names = vectorizer.get_feature_names_out()
        
        # Get feature importances
        importances = pipeline['classifier'].feature_importances_
        
        # Sort feature importances
        indices = np.argsort(importances)[::-1]
        
        # Get top 50 features
        top_n = min(50, len(feature_names))
        top_features = [(feature_names[indices[i]], importances[indices[i]]) for i in range(top_n)]
        
        # Create DataFrame
        importance_df = pd.DataFrame(top_features, columns=['feature', 'importance'])
        
        # Save to CSV
        importance_df.to_csv(os.path.join(save_dir, 'feature_importance.csv'), index=False)
        
        # Plot feature importances
        plt.figure(figsize=(12, 10))
        sns.barplot(y='feature', x='importance', data=importance_df.head(20))
        plt.title('Top 20 Important Features')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_importance.png'))
        
        print(f"\nFeature importance analysis saved to {save_dir}")
        return importance_df
    else:
        print("Model doesn't have feature_importances_ attribute. Skipping feature importance analysis.")
        return None

def main():
    """
    Main function to run optimized model training and analysis
    """
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('analysis', exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_df, val_df = load_and_preprocess_data(
        'data/twitter_training.csv',
        'data/twitter_validation.csv',
        remove_stopwords=True,
        lemmatization=True
    )
    
    # Get text and labels
    X_train = train_df['processed_tweet']
    y_train = train_df['sentiment']
    X_val = val_df['processed_tweet']
    y_val = val_df['sentiment']
    
    # Create and train optimized pipeline
    pipeline = create_optimized_pipeline()
    pipeline, y_pred, accuracy, report = train_and_evaluate(pipeline, X_train, y_train, X_val, y_val)
    
    # Save the model
    model_path = save_optimized_model(pipeline)
    
    # Analyze errors
    print("\nAnalyzing misclassifications...")
    misclassified_df, error_matrix = analyze_errors(X_val, y_val, y_pred)
    
    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    importance_df = analyze_feature_importance(pipeline)
    
    print("\nOptimized model training and analysis complete!")

if __name__ == "__main__":
    main() 