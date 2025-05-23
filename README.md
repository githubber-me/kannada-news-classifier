# Kannada News Headline Classifier

This Streamlit app classifies Kannada news headlines into categories like Sports, Entertainment, and Technology using a combination of a local Naive Bayes model and Google's Gemini API.

## Local Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your Gemini API key: `GEMINI_API_KEY=your_key_here`
4. Make sure you have the `train.csv` dataset file in the root directory
5. Run the app: `streamlit run app.py`

## Streamlit Cloud Deployment

1. Push this code to a GitHub repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app, selecting this GitHub repository
4. In the Advanced Settings:
   - Add your Gemini API key as a secret with the name `GEMINI_API_KEY`
   - Set the Python version to 3.9 or higher
5. Deploy the app

## Important Notes

- The app requires the `train.csv` file for model training
- For Streamlit Cloud deployment, you'll need to upload the dataset or modify the code to download it
- The model files will be created during the first run

## Features

- **Dataset Visualization**: View statistics and sample headlines from the training data
- **Model Performance**: Check the accuracy and other metrics for each category
- **Live Classification**: Enter or select a Kannada headline and get instant classification
- **Confidence Scores**: See probability scores for each possible category
- **Hybrid Ensemble Model**: Uses a weighted ensemble of Google's Gemini API and a local Naive Bayes model
- **Model Persistence**: Save and load trained models from disk

## Requirements

- Python 3.9 or earlier (recommended due to `inltk` compatibility)
- Required libraries:
  - streamlit
  - pandas
  - scikit-learn
  - inltk
  - google-generativeai
  - python-dotenv
  - other dependencies installed automatically
- Google Gemini API key

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd kannada-nlp
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. **Important Note**: If you're using Python 3.10+, you need to manually patch the fastai library (which inltk depends on):
   
   Edit the file:
   ```
   <python-path>/site-packages/fastai/imports/core.py
   ```
   
   Change:
   ```python
   from collections import abc, Counter, defaultdict, Iterable, namedtuple, OrderedDict
   ```
   
   To:
   ```python
   from collections import abc, Counter, defaultdict, namedtuple, OrderedDict
   from collections.abc import Iterable
   ```

4. Create a `.env` file in the project directory and add your Google Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

1. Make sure the training data `train.csv` is in the project directory
2. Ensure your `.env` file contains a valid Gemini API key
3. Run the application:
   ```
   streamlit run app.py
   ```
4. A browser window will open with the application
5. The first time you run the app, it will download the necessary Kannada language models and train a local model

## Hybrid Model Architecture

The application uses a weighted ensemble approach:
- **Local Model**: A Naive Bayes classifier trained on your Kannada news dataset
- **Google Gemini 1.5 Flash**: A state-of-the-art language model that processes Kannada text
- **Weighted Combination**: The predictions from both models are combined to produce the final classification

## Dataset Format

The application expects a CSV file named `train.csv` with at least these columns:
- `headline`: The Kannada news headline text
- `label`: The category label (e.g., "sports", "entertainment", "tech")
- [Current Dataset](https://www.kaggle.com/datasets/disisbig/kannada-news-dataset)

## Project Structure

- `app.py`: The main Streamlit application with model training, prediction, and UI
- `train.csv`: Training dataset with Kannada headlines and their categories
- `model/`: Directory containing the saved models and metadata
- `README.md`: Documentation (this file)
- `.env`: Environment file containing the Gemini API key (not tracked in git)

## Author

- This project is built by [Anthahkarana](github.com/githubber-me)