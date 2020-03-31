### Pandas
import pandas as pd
import numpy as np
### Regex
import re
### Import nltk
import nltk
nltk.download('punkt')
nltk.download('wordnet')
### Normalisation / Tokenisation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
### Stemming & Lemming
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
### Feature Union
from sklearn.pipeline import Pipeline, FeatureUnion
### Custom Transformer
from sklearn.base import BaseEstimator, TransformerMixin
### Model training
from sklearn.model_selection import train_test_split
### Pipeline test results
from sklearn.metrics import confusion_matrix
### Pipeline transformation
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
### GridSearch
from sklearn.model_selection import train_test_split, GridSearchCV
### Output Classifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
### SQL Alchemy
import sqlalchemy
### SQL
import sqlite3
### Pickle
import pickle

# load data from database
conn = sqlite3.connect('InsertDatabaseName.db')
df = pd.read_sql('SELECT * FROM InsertTableName', con = conn)
categories = df.columns.drop(['id', 'message', 'original', 'genre'])

X = df.message.values
y = df[categories]

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
	
knn = KNeighborsClassifier(n_neighbors=3)

pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(knn))
])

X_train, X_test, y_train, y_test = train_test_split(X, y)
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

accuracy = (y_pred == y_test).mean()
accuracy

parameters = {
    'vect__analyzer': ('word', 'sentence'),   
    'clf__estimator__n_neighbors': [3, 10],
    'clf__estimator__p': [2, 5]
}

cv = GridSearchCV(pipeline, param_grid=parameters)
pipeline.get_params()

cv.fit(X_train, y_train)
y_pred = model.predict(X_test)

def display_results(cv, y_test, y_pred):
    labels = np.unique(y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = (y_pred == y_test).mean()

    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
    print("\nBest Parameters:", cv.best_params_)
    
display_results(cv, y_test, y_pred)

