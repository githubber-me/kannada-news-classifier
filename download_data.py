import os
import pandas as pd
import streamlit as st

def get_dataset():
    """
    This function checks if the dataset exists, and if not, 
    creates a small sample dataset for demonstration purposes.
    
    In a production scenario, you would download the dataset from 
    a permanent storage location (like AWS S3, Google Drive, etc.)
    """
    if os.path.exists('train.csv'):
        print("Dataset found locally.")
        return True
    
    print("Dataset not found. Creating sample dataset...")
    
    # Create a sample dataset for demonstration
    sample_data = {
        'headline': [
            'ಕ್ರಿಕೆಟ್ ಪಂದ್ಯದಲ್ಲಿ ಭಾರತ ಗೆಲುವು',
            'ಹೊಸ ಕನ್ನಡ ಚಿತ್ರ ನಿರ್ದೇಶಕರು ಘೋಷಣೆ',
            'ಆಪಲ್ ಹೊಸ ಐಫೋನ್ ಬಿಡುಗಡೆ ಮಾಡಿದೆ',
            'ವಿರಾಟ್ ಕೊಹ್ಲಿ ಭರ್ಜರಿ ಶತಕ',
            'ಸಿನಿಮಾ ಬಿಡುಗಡೆಗೆ ಸಿದ್ಧವಾಗಿದೆ',
            'ಮೊಬೈಲ್ ಫೋನ್ ದರಗಳು ಕಡಿಮೆಯಾಗಿದೆ',
        ],
        'label': [
            'sports',
            'entertainment',
            'technology',
            'sports',
            'entertainment',
            'technology',
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('train.csv', index=False)
    print("Sample dataset created.")
    return True

if __name__ == "__main__":
    get_dataset() 