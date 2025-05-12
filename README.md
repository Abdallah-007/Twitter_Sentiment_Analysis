# Twitter Sentiment Analysis

This project implements sentiment analysis on Twitter data using NLP and machine learning techniques. The system classifies tweets into four sentiment categories: Positive, Negative, Neutral, and Irrelevant.

![Streamlit Demo](https://i.imgur.com/PmbJxoO.png)

## Project Structure
- `data/`: Contains the Twitter datasets
- `src/`: Python source code for the sentiment analysis
  - `preprocess.py`: Text preprocessing utilities
  - `model.py`: Basic ML model implementation
  - `model_tuning.py`: Hyperparameter tuning and model comparison
  - `visualization.py`: Data visualization scripts
  - `optimized_model.py`: Optimized model implementation with error analysis
  - `predict.py`: Prediction utilities
  - `streamlit_app.py`: Interactive web application using Streamlit
- `models/`: Saved ML models
- `visualizations/`: Generated visualizations
- `analysis/`: Model analysis results

## Key Results

After comparing various vectorization methods and classifiers, we found that:

1. **Best Vectorization Method**: N-gram Bag of Words (accuracy: 90.9%)
2. **Best Classifier**: Random Forest (accuracy: 97.0%)
3. **Optimized Model Performance**: 92.6% accuracy on validation set
   - Precision: 0.93, Recall: 0.92, F1-score: 0.93

Class-specific performance:
- Irrelevant: 92% F1-score
- Negative: 92% F1-score
- Neutral: 93% F1-score
- Positive: 93% F1-score

## Setup

### Local Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/twitter-sentiment-analysis.git
   cd twitter-sentiment-analysis
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   
   # On Linux/Mac
   source venv/bin/activate
   
   # On Windows
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Training the Model

```
python src/optimized_model.py
```

This script:
- Trains an optimized Random Forest model with n-gram features
- Evaluates model performance
- Analyzes misclassifications
- Identifies important features
- Saves the model for future use

## Usage

### Data Exploration and Visualization
```
python src/visualization.py
```
This generates various visualizations in the `visualizations/` directory:
- Word clouds for each sentiment
- Tweet length distributions
- Top words by sentiment
- Sentiment and topic distributions
- Dimension reduction visualizations (PCA, t-SNE)

### Model Tuning
```
python src/model_tuning.py
```
This script:
- Compares different vectorization methods (BoW, TF-IDF, n-grams, etc.)
- Compares different classifiers (Naive Bayes, Logistic Regression, SVM, Random Forest, etc.)
- Performs hyperparameter tuning for the best classifier

### Streamlit Web App

Run the interactive Streamlit web application:

```
streamlit run src/streamlit_app.py
```

Features:
- Single text analysis with sentiment and confidence display
- Batch analysis with file upload 
- Live sentiment analysis simulation
- Word clouds and visualizations
- Model information and feature importance display
- Download results as CSV

## Deployment

### Deploy to Streamlit Cloud

1. Sign up for a free account at [Streamlit Cloud](https://streamlit.io/cloud)

2. Create a GitHub repository with your code:
   ```
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/twitter-sentiment-analysis.git
   git push -u origin main
   ```

3. Connect your GitHub repository to Streamlit Cloud
   - Click "New app"
   - Select your repository, branch, and `src/streamlit_app.py` as the main file
   - Click "Deploy"

### Deploy to Heroku

1. Create a `Procfile` in the root directory with the following content:
   ```
   web: sh setup.sh && streamlit run src/streamlit_app.py
   ```

2. Create a `setup.sh` file:
   ```bash
   mkdir -p ~/.streamlit/
   echo "\
   [general]\n\
   email = \"your-email@example.com\"\n\
   " > ~/.streamlit/credentials.toml
   echo "\
   [server]\n\
   headless = true\n\
   enableCORS = false\n\
   port = $PORT\n\
   " > ~/.streamlit/config.toml
   ```

3. Deploy to Heroku:
   ```
   heroku create
   git push heroku main
   ```

## Feature Importance

The top features (words) influencing sentiment classification are:
1. love
2. fix
3. game
4. fuck
5. server
6. shit
7. best
8. good
9. fun
10. hate

## Error Analysis

The model performs best on Positive sentiment (96.8% accuracy) and worst on Irrelevant sentiment (86.6% accuracy).

Common misclassification patterns:
- Neutral tweets incorrectly classified as Negative
- Irrelevant tweets incorrectly classified as Positive
- Neutral tweets incorrectly classified as Positive

## Future Improvements

1. Add more preprocessing steps (spelling correction, emoji handling, etc.)
2. Explore ensemble methods combining different model types
3. Implement cross-validation for more robust performance estimates
4. Add live Twitter API integration for real-time analysis

## License

MIT

## Acknowledgments

- Dataset provided by [source_name]
- Inspired by [related_research] 