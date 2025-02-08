import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

# Preprocess text
# (tokenization, stopwords removal, lemmatization)
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


# Create sentiment analyzer
def get_sentiment(text):
    scores = analyzer.polarity_scores(text)

    sentiment = 1 if scores['pos'] > 0 else 0

    return sentiment


if __name__ == "__main__":
    # download nltk corpus (first time only)
    # nltk.download('all')

    df = pd.read_csv('YoutubeCommentsDataSet_Filtered.csv')

    df['Comment'] = df["Comment"].apply(preprocess_text)

    df['sentiment'] = df['Comment'].apply(get_sentiment)

    print(df.head())
