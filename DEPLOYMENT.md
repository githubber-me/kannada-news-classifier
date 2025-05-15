# Deploying the Kannada News Classifier to Streamlit Cloud

This guide provides step-by-step instructions for deploying the Kannada News Classifier app to Streamlit Cloud.

## Prerequisites

1. A GitHub account
2. A Google Gemini API key (optional, but recommended)

## Deployment Steps

### 1. Push Your Code to GitHub

First, create a new GitHub repository and push your code:

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/your-username/kannada-news-classifier.git
git push -u origin main
```

### 2. Sign Up for Streamlit Cloud

Go to [Streamlit Cloud](https://streamlit.io/cloud) and sign up using your GitHub account.

### 3. Deploy Your App

1. Click on "New app" in the Streamlit Cloud dashboard
2. Connect to your GitHub repository
3. Select the repository, branch (main), and the main file (app.py)
4. Click "Advanced settings" and add:
   - Set the Python version to 3.9 or higher
   - Add your Gemini API key as a secret with the name `GEMINI_API_KEY`
5. Click "Deploy"

### 4. Monitor Deployment

After clicking deploy, Streamlit Cloud will:
1. Clone your repository
2. Install the required dependencies from requirements.txt
3. Run your app
4. Deploy it to a public URL

This process usually takes 2-5 minutes.

## Troubleshooting

### Common Issues and Solutions

1. **Missing dependencies**: 
   - Make sure all required packages are listed in requirements.txt with the correct versions

2. **API Key errors**: 
   - Verify your Gemini API key is correctly set in the Streamlit Cloud secrets

3. **Dataset issues**: 
   - The app is configured to create a sample dataset if train.csv is not found
   - For a production app, consider storing your dataset in a cloud storage service and updating the code to download it

4. **Model training timeouts**: 
   - If your model training takes too long, Streamlit Cloud might time out
   - Consider preprocessing your data and including a pre-trained model in your repository

## Updating Your App

When you push changes to your GitHub repository, Streamlit Cloud will automatically redeploy your app with the new changes.

## Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-cloud) 