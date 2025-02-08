from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from typing import List
import pickle

import uvicorn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import nltk
from nltk.data import find


# Function to check and download necessary resources
def download_nltk_resources():
    resources = ['tokenizers/punkt', 'corpora/stopwords', 'corpora/wordnet']
    
    for resource in resources:
        try:
            find(resource)  # Check if resource exists
        except LookupError:
            print(f"Downloading: {resource} ...")
            nltk.download(resource.split('/')[-1])  # Download if not found

# Call the function once
download_nltk_resources()


vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
svc_model = pickle.load(open('svc_model.pkl', 'rb'))

'''
# Load the saved Support Vector Classifier (SVC) model
with open('svc_model.pkl', 'rb') as model_file:
    svc_model = pickle.load(model_file)

# Load the saved TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)
'''


app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development only)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)


# Function to preprocess input text
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Join the tokens back into a string
    preprocessed_text = " ".join(lemmatized_tokens)

    return preprocessed_text


class TextInput(BaseModel):
    input: str


@app.get('/')
def root():
    return {
        'description': 'this is sentiment analysis project.'
    }


@app.post('/predict')
def predict(input: TextInput):
    try:
        text = input.input

        text_preprocessed = preprocess_text(text)

        X_vec = vectorizer.transform([text_preprocessed])

        score = svc_model.predict(X_vec)

        # Map score to sentiment label
        score_to_val = {1: 'positive', 0: 'neutral', -1: 'negative'}
        sentiment = score_to_val[score[0]]

        return {
            'sentiment': sentiment
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == '__main__':
    uvicorn.run('app:app', host='127.0.0.1', port=8000, reload=True)
