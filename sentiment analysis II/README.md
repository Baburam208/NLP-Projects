## This sentiment analyzer project
The data used is: Youtube Comments Dataset from kaggle

link: https://www.kaggle.com/datasets/atifaliak/youtube-comments-dataset

There are three labels in the dataset: positive, neutral, and negative.

There are 18408 rows and 2 columns in the dataset.

## Exploratory data analysis
1. NaN values
Finding any missing values in the dataset.
```
# Get NaN counts for all columns
nan_counts = df.isna().sum()
print(nan_counts)
```

There are 44 nan value found in the 'Comment' column, which are removed.
```
df['Comment'].isnull().sum()
```

2. Handling wierd characters
We have some characters like: donâ€™t, doesnâ€™t, itâ€™s and so on. There are successfully converted to proper format.
```
# ftfy automatically detects and fixes common encoding mistakes.
from ftfy import fix_text

df2 = df['Comment'].apply(fix_text)
```

3. Remove comments in other languages except English
We have comments in other languages as well such as
* തന്നെ തളർത്താൻ നോക്കിയവർ പോലും അവന്റെ ഉയർച്ചയി...
* 火災があったのはラブホテルじゃん。。。
* මොකුත් කියන්නෑ සිංහ්ල්ලයට තේරෙන්නැති නිසා
* पत्रकार को बहुत जल्दी पता चलता है कि ak47 कहाँ...
* මේ ඝන ගෙඩියො ටික නම් එලවන්න ඕනෙ රටෙන්ම
* සජි ඔයගෙ තත්තා හිටියනම් ටැයර්එකෙ යවන්නෙ නෙද සජ...
and so on.

we don't went through their conversion to English. We simply remove them.

The following code snippet removes non-English characters from the column 'Comment'
```
# Detect and filter rows with non-ASCII characters
df_filtered2 = df[~df['Comment'].str.contains(r'[^\x00-\x7F]', regex=True, na=False)]
```

Then finally the filtered dataframe is saved as follows:
```
df_filtered2.to_csv("YoutubeCommentsDataSet_Filtered.csv", index=False)
```
Now, the filtered dataframe has 14314 rows.
## Model Training
### Preprocessing
In the preprocessing step, we first tokenize the text, then remove stopwords from the text, and finally converted to vector form.

We have used `TfidfVectorizer()` as well as `CountVectorizer()` and `TfidfTransformer()` to convert the text to the numerical form.

Using `TfidfVectorizer()`
```
# Convert text to TF-IDF vectors (using TfidfVectorizer)
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to top 5000 features
X = vectorizer.fit_transform(df['Comment'])
```
Or 
```
count_vectorizer = CountVectorizer(max_features=5000)

X_vec = count_vectorizer.fit_transform(df['Comment'])
X_vec = X_vec.todense()

tfidf_transformer = TfidfTransformer()
X_vec = tfidf_transformer.fit_transform(np.asarray(X_vec))
X_vec = X_vec.todense()
```

Labels are mapped as follows:
```
# Lets map the 'Sentiment' label to numerical format
# 'positive': 1, 'neutral': 0, and 'negative': -1

sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
df["Sentiment_Score"] = df["Sentiment"].map(sentiment_map)
```

We trained the model using the `SVM` and `Random Forest`.

`SVM` gives the best result (accuracy of 75%).

**We have not performed hyperparameter tuning in this project, this is our future work.**

### Pickling
Finally the trained model is pickled to use for the future. Both model and vectorizer are saved.

```
import pickle

pickle.dump(svc_model, open('svc_model', 'wb'))
pickle.dump(vectorizer, open('tfidf_vectorizer', 'wb'))
```

## Finally the model is served using API
We have created a API endpoint to serve the model for inference, using FastAPI.

Below is the screenshots of the model inference UI and server running.

![alt text](<screen shot for sentiment analyzer II.png>)


**The model is only 75 % accurate, so the predictor does not predict so well.**

**Note:**
* In the training set as well we found some weird comments,such as

    `linus this is not a table also linus lets use this as a chair` 
    
    Even though this is in English alphabet, but I don't think I convay any meaning.

* We conclude that such examples make mode less accurate, and we have curated 'Comments' probably increase model accuracy.