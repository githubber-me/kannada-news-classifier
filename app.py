import streamlit as st
import pandas as pd
import pickle
import os
import json
from dotenv import load_dotenv
import google.generativeai as genai

# Monkey patch for fastai in Python 3.10+ before importing inltk
import sys
if sys.version_info >= (3, 10):
    import importlib
    import types
    
    # Create a fake collections module with Iterable
    class FakeCollections(types.ModuleType):
        def __getattr__(self, name):
            if name == 'Iterable':
                from collections.abc import Iterable
                return Iterable
            from collections import __getattribute__
            return __getattribute__(name)
    
    # Replace the collections module in sys.modules
    sys.modules['collections'] = FakeCollections('collections')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from inltk.inltk import setup, tokenize
from download_data import get_dataset

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Kannada News Classifier",
    page_icon="📰",
    layout="wide"
)

# Title and description
st.title("Kannada News Headline Classifier")
st.markdown("""
This app classifies Kannada news headlines into categories like Sports, Entertainment, 
and Technology. Enter a Kannada headline below and click 'Classify' to see the prediction.
""")

# Define model paths
MODEL_PATH = "model/kannada_news_classifier.pkl"
VECTORIZER_PATH = "model/tfidf_vectorizer.pkl"
METRICS_PATH = "model/metrics.pkl"

# Initialize Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Try to get API key from Streamlit secrets if not found in environment variables
if not GEMINI_API_KEY and hasattr(st, 'secrets') and "GEMINI_API_KEY" in st.secrets:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
else:
    st.sidebar.warning("⚠️ Gemini API key not found in .env file or Streamlit secrets. Running with local model only.")
    gemini_model = None

# Create model directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

@st.cache_resource(show_spinner=True)
def setup_kannada_inltk():
    """Sets up iNLTK for the Kannada language and caches the result."""
    try:
        setup('kn')
        return True
    except Exception as e:
        st.error(f"Error setting up iNLTK for Kannada: {e}")
        st.error("Please ensure you have a working internet connection and necessary permissions.")
        return False

def preprocess_text(text):
    """Tokenizes Kannada text using iNLTK."""
    if pd.isna(text) or text == "":
        return ""
    try:
        tokens = tokenize(str(text), 'kn')
        return " ".join(tokens)
    except Exception as e:
        st.warning(f"Error tokenizing text: {e}")
        return "" # Return empty string on error to avoid pipeline breakage

@st.cache_data(show_spinner=True)
def load_data():
    """Loads and returns the dataset."""
    try:
        # Ensure dataset exists, if not create a sample one
        get_dataset()
        df = pd.read_csv('train.csv')
        return df
    except FileNotFoundError:
        st.error("Error: The file train.csv was not found. Please make sure it's in the correct directory.")
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

def save_model(model, vectorizer, metrics):
    """Save the model, vectorizer and metrics to disk."""
    try:
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        with open(VECTORIZER_PATH, 'wb') as f:
            pickle.dump(vectorizer, f)
        with open(METRICS_PATH, 'wb') as f:
            pickle.dump(metrics, f)
        return True
    except Exception as e:
        st.error(f"Error saving model to disk: {e}")
        return False

def load_saved_model():
    """Load the model, vectorizer and metrics from disk if they exist."""
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH) and os.path.exists(METRICS_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            with open(VECTORIZER_PATH, 'rb') as f:
                vectorizer = pickle.load(f)
            with open(METRICS_PATH, 'rb') as f:
                metrics = pickle.load(f)
            return model, vectorizer, metrics.get('accuracy'), metrics.get('report'), metrics.get('labels')
        except Exception as e:
            st.warning(f"Error loading saved model: {e}. Will retrain model.")
            return None, None, None, None, None
    return None, None, None, None, None

def train_model(df):
    """Trains and returns the model and vectorizer."""
    # Data preprocessing
    df_clean = df.copy()
    df_clean.dropna(subset=['headline', 'label'], inplace=True)
    
    # Ensure all text is string and handle potential NaNs
    df_clean['headline'] = df_clean['headline'].astype(str).fillna('')
    df_clean['processed_text'] = df_clean['headline'].apply(preprocess_text)
    
    # Filter out rows where processed_text might be empty
    df_clean = df_clean[df_clean['processed_text'].str.strip().astype(bool)]
    
    if df_clean.empty:
        return None, None, None, None, None
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df_clean['processed_text'], df_clean['label'], test_size=0.2, random_state=42
    )
    
    # TF-IDF vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # Train Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate model (for display purposes)
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    labels = df_clean['label'].unique()
    
    # Save metrics for later use
    metrics = {
        'accuracy': accuracy,
        'report': report,
        'labels': labels
    }
    
    # Save model to disk
    save_model(model, tfidf_vectorizer, metrics)
    
    return model, tfidf_vectorizer, accuracy, report, labels

def get_model_and_metrics(df):
    """Load model from disk if available, otherwise train a new model."""
    # Try to load the model first
    model, vectorizer, accuracy, report, labels = load_saved_model()
    
    # If loading failed, train a new model
    if model is None or vectorizer is None:
        with st.spinner("Training model for the first time... this might take a few minutes"):
            model, vectorizer, accuracy, report, labels = train_model(df)
    else:
        st.success("✅ Loaded pre-trained model from disk")
    
    return model, vectorizer, accuracy, report, labels

def predict_with_local_model(text, model, vectorizer):
    """Predicts the category of a Kannada headline using local model."""
    if not text.strip():
        return None, {}
    
    processed_text = preprocess_text(text)
    if not processed_text.strip():
        return None, {}
    
    text_tfidf = vectorizer.transform([processed_text])
    prediction = model.predict(text_tfidf)
    probabilities = model.predict_proba(text_tfidf)[0]
    
    # Get class probabilities
    class_probs = {}
    for i, label in enumerate(model.classes_):
        class_probs[label] = float(probabilities[i])  # Convert to float for JSON serialization
    
    return prediction[0], class_probs

def predict_with_gemini(text, available_categories):
    """Predicts the category of a headline using Gemini API."""
    if not gemini_model or not text.strip():
        return None, {}
    
    try:
        # Create prompt for Gemini
        prompt = f"""
        Classify the following Kannada news headline into one of these categories: {', '.join(available_categories)}.
        
        Headline: "{text}"
        
        Please analyze the headline and respond with a JSON object in the following format:
        {{
            "category": "the_predicted_category",
            "confidence_scores": {{
                "category1": score1,
                "category2": score2,
                ...
            }},
            "explanation": "A brief explanation of why this category was chosen"
        }}
        
        Scores should sum to 1.0 and represent your confidence in each category.
        """
        
        response = gemini_model.generate_content(prompt)
        response_text = response.text
        
        # Extract JSON from response
        try:
            # Try to find JSON in the response (it might be wrapped in markdown code blocks)
            if "```json" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_text = response_text.split("```")[1].strip()
            else:
                json_text = response_text
            
            result = json.loads(json_text)
            
            # Ensure we have the expected structure
            if "category" not in result or "confidence_scores" not in result:
                raise ValueError("Missing required fields in Gemini response")
                
            return result["category"], result["confidence_scores"]
        except (json.JSONDecodeError, ValueError, IndexError) as e:
            st.warning(f"Error parsing Gemini response: {e}. Using local model only.")
            return None, {}
    except Exception as e:
        st.warning(f"Error querying Gemini API: {e}. Using local model only.")
        return None, {}

def predict_headline(text, model, vectorizer):
    """Predicts the category of a headline combining local model and Gemini (80-20 split)."""
    if not text.strip():
        return None, "Empty headline provided."
    
    # Get available categories from local model
    available_categories = model.classes_.tolist()
    
    # Get predictions from both models
    local_prediction, local_probs = predict_with_local_model(text, model, vectorizer)
    gemini_prediction, gemini_probs = predict_with_gemini(text, available_categories)
    
    # If either prediction failed, return the other one
    if not local_prediction:
        return gemini_prediction, gemini_probs if gemini_prediction else "Could not process headline."
    
    if not gemini_prediction:
        return local_prediction, local_probs
    
    # Combine probabilities with 80% weight to Gemini and 20% to local model
    combined_probs = {}
    
    # Initialize with zeros for all categories
    for category in available_categories:
        combined_probs[category] = 0.0
    
    # Add local model probabilities (20% weight)
    for category, prob in local_probs.items():
        combined_probs[category] += 0.2 * prob
    
    # Add Gemini probabilities (80% weight)
    for category, prob in gemini_probs.items():
        if category in combined_probs:  # Make sure category exists in our model
            combined_probs[category] += 0.8 * float(prob)
    
    # Find the highest probability category
    final_prediction = max(combined_probs.items(), key=lambda x: x[1])[0]
    
    # Normalize probabilities to sum to 1
    prob_sum = sum(combined_probs.values())
    if prob_sum > 0:
        for category in combined_probs:
            combined_probs[category] /= prob_sum
    
    return final_prediction, combined_probs

# Main app workflow
def main():
    # Set up iNLTK (cached)
    setup_success = setup_kannada_inltk()
    if not setup_success:
        st.stop()
    
    # Create tabs for different app sections
    input_tab, analysis_tab = st.tabs(["📝 Classify Headlines", "📊 Model Performance"])
    
    # Load data (cached)
    df = load_data()
    if df is None:
        st.stop()
    
    # Display dataset stats in sidebar
    with st.sidebar:
        st.header("Dataset Information")
        st.write(f"Total headlines: {len(df)}")
        st.write("Category distribution:")
        category_counts = df['label'].value_counts()
        st.dataframe(category_counts)
        
        st.header("Sample Headlines")
        st.dataframe(df.head(5)[['headline', 'label']])
        
        # Add option to retrain model
        st.header("Model Options")
        if st.button("🔄 Retrain Model", help="Force retrain the model even if a saved model exists"):
            # Delete existing models to force retraining
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            if os.path.exists(VECTORIZER_PATH):
                os.remove(VECTORIZER_PATH)
            if os.path.exists(METRICS_PATH):
                os.remove(METRICS_PATH)
            st.experimental_rerun()
        
        # Display model weightage information
        st.header("Model Weightage")
        st.info("🤖 This classifier uses a weighted combination of models:")
        st.markdown("""
        - **Google Gemini AI**: 80% weight
        - **Local Naive Bayes**: 20% weight
        """)
        
        if not gemini_model:
            st.warning("⚠️ Gemini API not configured. Using local model only.")
    
    # Get model (either load from disk or train new)
    model, vectorizer, accuracy, report, labels = get_model_and_metrics(df)
    
    if model is None or vectorizer is None:
        st.error("Model training failed. Please check the data and logs.")
        st.stop()
    
    # INPUT TAB - User Input and Classification
    with input_tab:
        st.header("Enter a Kannada News Headline to Classify")
        st.markdown("""
        Type or paste a Kannada news headline in the text box below to classify it.
        The model will predict whether it belongs to sports, entertainment, technology, or other categories.
        """)
        
        # Sample headlines to try
        with st.expander("📋 Click here to see sample headlines you can try"):
            sample_headlines = {
                "Sports": [
                    "ಕ್ರಿಕೆಟ್ ಪಂದ್ಯದಲ್ಲಿ ಭಾರತ ಗೆಲುವು",  # Cricket match victory for India
                    "ವಿರಾಟ್ ಕೊಹ್ಲಿ ಭರ್ಜರಿ ಶತಕ",  # Virat Kohli's century
                    "ಟೀಂ ಇಂಡಿಯಾ ಆಟಗಾರರ ಹರಾಜು"  # Team India players auction
                ],
                "Entertainment": [
                    "ಹೊಸ ಕನ್ನಡ ಚಿತ್ರ ನಿರ್ದೇಶಕರು ಘೋಷಣೆ",  # New Kannada film director announced
                    "ಸಿನಿಮಾ ಬಿಡುಗಡೆಗೆ ಸಿದ್ಧವಾಗಿದೆ",  # Movie ready for release
                    "ನಟಿಯ ಹೊಸ ಚಿತ್ರ ಘೋಷಣೆ"  # Actress's new film announced
                ],
                "Tech": [
                    "ಆಪಲ್ ಹೊಸ ಐಫೋನ್ ಬಿಡುಗಡೆ ಮಾಡಿದೆ",  # Apple released new iPhone
                    "ಮೊಬೈಲ್ ಫೋನ್ ದರಗಳು ಕಡಿಮೆಯಾಗಿದೆ",  # Mobile phone prices decreased
                    "ಹೊಸ ತಂತ್ರಜ್ಞಾನ ಅಭಿವೃದ್ಧಿ"  # New technology development
                ]
            }
            
            for category, headlines in sample_headlines.items():
                st.subheader(f"{category} Headlines:")
                for headline in headlines:
                    if st.button(headline, key=f"sample_{headline[:20]}"):
                        st.session_state["user_headline"] = headline
        
        # Initialize session state for user headline if not exists
        if "user_headline" not in st.session_state:
            st.session_state["user_headline"] = ""
            
        # User input text area
        user_input = st.text_area(
            "Enter a Kannada news headline to classify:",
            value=st.session_state["user_headline"],
            height=100,
            key="user_headline"
        )
        
        # Buttons row
        col1, col2 = st.columns([1, 4])
        with col1:
            # Clear button to reset the text area
            if st.button("🧹 Clear", key="clear_button"):
                st.session_state["user_headline"] = ""
                st.experimental_rerun()
                
        with col2:
            # Classification button with prominent styling
            classify_button = st.button("🔍 Classify Headline", type="primary", use_container_width=True)
        
        # Perform classification when button is clicked
        if classify_button:
            if user_input:
                with st.spinner("Classifying the headline..."):
                    prediction, probabilities = predict_headline(user_input, model, vectorizer)
                    
                    if isinstance(probabilities, dict):
                        # Display prediction with confidence in a colorful box
                        st.markdown(f"""
                        <div style="padding: 20px; border-radius: 10px; background-color: #f0f8ff; margin-top: 20px;">
                            <h3 style="color: #0066cc;">📰 Prediction Result</h3>
                            <p style="font-size: 1.2rem; font-weight: bold;">"{user_input}"</p>
                            <p style="font-size: 1.5rem; color: #2e8b57;">Category: {prediction.upper()}</p>
                            <p style="font-size: 0.9rem; color: #666;">Using a weighted ensemble of models!</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display probability distribution
                        st.subheader("Confidence Scores")
                        
                        # Sort probabilities
                        sorted_probs = {k: v for k, v in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)}
                        
                        # Display as horizontal bars
                        for category, prob in sorted_probs.items():
                            # Highlight the predicted category
                            if category == prediction:
                                st.markdown(f"**{category.upper()}**: {prob:.2%}")
                                st.progress(float(prob))
                            else:
                                st.markdown(f"{category.capitalize()}: {prob:.2%}")
                                st.progress(float(prob))
                    else:
                        # Display error message
                        st.error(probabilities)
            else:
                st.warning("Please enter a headline before classifying.")
    
    # ANALYSIS TAB - Model Performance Details
    with analysis_tab:
        st.header("Local Model Performance")
        st.write(f"🎯 Overall Model Accuracy: {accuracy:.2%}")
        st.info("Note: These metrics are for the local Naive Bayes model only. The combined model with Gemini AI may have different performance characteristics.")
        
        # Create sub-tabs for different categories
        performance_tabs = st.tabs([f"📊 {label.capitalize()} Performance" for label in sorted(labels)])
        
        for i, label in enumerate(sorted(labels)):
            with performance_tabs[i]:
                metrics = report.get(label, {})
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Precision", f"{metrics.get('precision', 0):.2%}")
                with col2:
                    st.metric("Recall", f"{metrics.get('recall', 0):.2%}")
                with col3:
                    st.metric("F1-Score", f"{metrics.get('f1-score', 0):.2%}")
                
                # Explanation of metrics
                st.markdown("""
                **What do these metrics mean?**
                - **Precision**: Percentage of headlines classified as this category that are actually correct
                - **Recall**: Percentage of actual headlines from this category that were correctly identified
                - **F1-Score**: Balance between precision and recall (harmonic mean)
                """)
                
                # Display confusion examples if available
                st.markdown("### Common Prediction Patterns")
                st.write("Categories most commonly confused with this one:")
                
                # This would ideally be populated with actual confusion matrix data
                # We're using placeholder text for now
                st.info(f"The model sometimes confuses {label} headlines with other categories when they contain similar keywords or themes.")

# Run the app
if __name__ == "__main__":
    main() 
