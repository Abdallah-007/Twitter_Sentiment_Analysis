"""
Twitter Sentiment Analysis - Model Tuning and Comparison
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Import local modules
from preprocess import load_and_preprocess_data
from model import vectorize_text, save_model

def compare_vectorizers(X_train, X_val, y_train, y_val, classifier=LogisticRegression(max_iter=1000)):
    """
    Compare different text vectorization methods
    
    Args:
        X_train: Training text data
        X_val: Validation text data
        y_train: Training labels
        y_val: Validation labels
        classifier: ML classifier to use
    
    Returns:
        dict: Results comparing different vectorization methods
    """
    results = {}
    
    # 1. Bag of Words (CountVectorizer)
    print("\n1. Bag of Words Vectorization")
    count_vec = CountVectorizer(max_features=5000)
    X_train_bow = count_vec.fit_transform(X_train)
    X_val_bow = count_vec.transform(X_val)
    
    bow_model = classifier.fit(X_train_bow, y_train)
    bow_pred = bow_model.predict(X_val_bow)
    bow_acc = accuracy_score(y_val, bow_pred)
    print(f"Accuracy: {bow_acc:.4f}")
    
    results['bow'] = {
        'vectorizer': count_vec,
        'model': bow_model,
        'accuracy': bow_acc,
        'predictions': bow_pred
    }
    
    # 2. TF-IDF Vectorization
    print("\n2. TF-IDF Vectorization")
    tfidf_vec = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vec.fit_transform(X_train)
    X_val_tfidf = tfidf_vec.transform(X_val)
    
    tfidf_model = classifier.fit(X_train_tfidf, y_train)
    tfidf_pred = tfidf_model.predict(X_val_tfidf)
    tfidf_acc = accuracy_score(y_val, tfidf_pred)
    print(f"Accuracy: {tfidf_acc:.4f}")
    
    results['tfidf'] = {
        'vectorizer': tfidf_vec,
        'model': tfidf_model,
        'accuracy': tfidf_acc,
        'predictions': tfidf_pred
    }
    
    # 3. N-gram Bag of Words
    print("\n3. N-gram Bag of Words (1-3 grams)")
    ngram_bow_vec = CountVectorizer(max_features=10000, ngram_range=(1, 3))
    X_train_ngram_bow = ngram_bow_vec.fit_transform(X_train)
    X_val_ngram_bow = ngram_bow_vec.transform(X_val)
    
    ngram_bow_model = classifier.fit(X_train_ngram_bow, y_train)
    ngram_bow_pred = ngram_bow_model.predict(X_val_ngram_bow)
    ngram_bow_acc = accuracy_score(y_val, ngram_bow_pred)
    print(f"Accuracy: {ngram_bow_acc:.4f}")
    
    results['ngram_bow'] = {
        'vectorizer': ngram_bow_vec,
        'model': ngram_bow_model,
        'accuracy': ngram_bow_acc,
        'predictions': ngram_bow_pred
    }
    
    # 4. Character-level n-grams
    print("\n4. Character N-gram TF-IDF")
    char_vec = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), max_features=20000)
    X_train_char = char_vec.fit_transform(X_train)
    X_val_char = char_vec.transform(X_val)
    
    char_model = classifier.fit(X_train_char, y_train)
    char_pred = char_model.predict(X_val_char)
    char_acc = accuracy_score(y_val, char_pred)
    print(f"Accuracy: {char_acc:.4f}")
    
    results['char_tfidf'] = {
        'vectorizer': char_vec,
        'model': char_model,
        'accuracy': char_acc,
        'predictions': char_pred
    }
    
    # 5. Combined word and character n-grams
    print("\n5. Combined Word and Character N-grams")
    # We'll need to combine predictions from both models
    combined_pred = []
    for w_pred, c_pred in zip(tfidf_pred, char_pred):
        # Simple ensemble: if word and char predictions match, use that; otherwise use the word prediction
        if w_pred == c_pred:
            combined_pred.append(w_pred)
        else:
            combined_pred.append(w_pred)
    
    combined_acc = accuracy_score(y_val, combined_pred)
    print(f"Accuracy: {combined_acc:.4f}")
    
    results['combined'] = {
        'accuracy': combined_acc,
        'predictions': combined_pred
    }
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    methods = ['Bag of Words', 'TF-IDF', 'N-gram BoW', 'Char N-gram', 'Combined']
    accuracies = [bow_acc, tfidf_acc, ngram_bow_acc, char_acc, combined_acc]
    
    sns.barplot(x=methods, y=accuracies)
    plt.title('Comparison of Vectorization Methods')
    plt.ylabel('Accuracy')
    plt.xlabel('Method')
    plt.ylim(0.5, 1.0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('models/vectorization_comparison.png')
    
    # Find the best vectorization method
    best_method = max([(k, v['accuracy']) for k, v in results.items() if 'accuracy' in v], key=lambda x: x[1])[0]
    print(f"\nBest vectorization method: {best_method}")
    
    return results, best_method

def compare_classifiers(X_train, X_val, y_train, y_val, vectorizer_type='tfidf'):
    """
    Compare different classifiers for sentiment analysis
    
    Args:
        X_train: Training text data
        X_val: Validation text data
        y_train: Training labels
        y_val: Validation labels
        vectorizer_type: Type of vectorizer to use ('tfidf' or 'bow')
    
    Returns:
        dict: Results comparing different classifiers
    """
    # Vectorize the data
    if vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=5000)
    else:
        vectorizer = CountVectorizer(max_features=5000)
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    
    # Define classifiers to compare
    classifiers = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Linear SVM': LinearSVC(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    # Train and evaluate each classifier
    results = {}
    accuracies = []
    
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        
        # Train the model
        clf.fit(X_train_vec, y_train)
        
        # Make predictions
        y_pred = clf.predict(X_val_vec)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_val, y_pred)
        accuracies.append(accuracy)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(y_val, y_pred))
        
        # Save results
        results[name] = {
            'model': clf,
            'accuracy': accuracy,
            'predictions': y_pred
        }
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(classifiers.keys()), y=accuracies)
    plt.title('Comparison of Classifiers')
    plt.ylabel('Accuracy')
    plt.xlabel('Classifier')
    plt.ylim(0.5, 1.0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('models/classifier_comparison.png')
    
    # Find the best classifier
    best_classifier = max([(k, v['accuracy']) for k, v in results.items()], key=lambda x: x[1])[0]
    print(f"\nBest classifier: {best_classifier}")
    
    return results, vectorizer, best_classifier

def tune_hyperparameters(X_train, X_val, y_train, y_val, classifier_type='lr'):
    """
    Tune hyperparameters for a specific classifier
    
    Args:
        X_train: Training text data
        X_val: Validation text data
        y_train: Training labels
        y_val: Validation labels
        classifier_type: Type of classifier to tune ('nb', 'lr', 'svm', 'rf', 'gb')
    
    Returns:
        tuple: (best_estimator, best_params, best_score)
    """
    # Vectorize the text
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    
    # Define parameter grids for each classifier
    if classifier_type == 'nb':
        model = MultinomialNB()
        param_grid = {
            'alpha': [0.01, 0.1, 0.5, 1.0, 2.0]
        }
        
    elif classifier_type == 'lr':
        model = LogisticRegression(max_iter=1000, random_state=42)
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
        
    elif classifier_type == 'svm':
        model = LinearSVC(random_state=42)
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'loss': ['hinge', 'squared_hinge'],
            'dual': [True, False]
        }
        
    elif classifier_type == 'rf':
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
    elif classifier_type == 'gb':
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        }
    
    else:
        raise ValueError("Invalid classifier type. Must be 'nb', 'lr', 'svm', 'rf', or 'gb'")
    
    # Perform grid search
    print(f"\nPerforming grid search for {classifier_type}...")
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_vec, y_train)
    
    # Get the best parameters
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"Best cross-validation score: {best_score:.4f}")
    print(f"Best parameters: {best_params}")
    
    # Evaluate on validation set
    best_model = grid_search.best_estimator_
    val_score = best_model.score(X_val_vec, y_val)
    print(f"Validation accuracy: {val_score:.4f}")
    
    # Save the best model
    save_model(best_model, vectorizer, prefix=f'tuned_{classifier_type}')
    
    return best_model, vectorizer, best_params, val_score

def main():
    """
    Main function to run all comparisons and tuning
    """
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
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
    
    # 1. Compare vectorization methods
    print("\n===== Comparing Vectorization Methods =====")
    vec_results, best_vec_method = compare_vectorizers(X_train, X_val, y_train, y_val)
    
    # 2. Compare classifiers
    print("\n===== Comparing Classifiers =====")
    clf_results, vectorizer, best_classifier = compare_classifiers(X_train, X_val, y_train, y_val)
    
    # 3. Tune hyperparameters for the best classifier
    print("\n===== Tuning Hyperparameters =====")
    classifier_mapping = {
        'Naive Bayes': 'nb',
        'Logistic Regression': 'lr',
        'Linear SVM': 'svm',
        'Random Forest': 'rf',
        'Gradient Boosting': 'gb'
    }
    
    best_clf_type = classifier_mapping.get(best_classifier, 'lr')
    best_model, best_vectorizer, best_params, best_score = tune_hyperparameters(
        X_train, X_val, y_train, y_val, classifier_type=best_clf_type
    )
    
    print("\n===== All comparisons and tuning complete =====")
    print(f"Best vectorization method: {best_vec_method}")
    print(f"Best classifier: {best_classifier}")
    print(f"Best hyperparameters: {best_params}")
    print(f"Best validation accuracy: {best_score:.4f}")
    
if __name__ == "__main__":
    main() 