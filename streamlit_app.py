"""
Twitter Sentiment Analysis - Entry point for Streamlit Cloud deployment
"""

# Make sure the src directory is in the path
import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Now import and run the main Streamlit app from the src directory
import src.streamlit_app 