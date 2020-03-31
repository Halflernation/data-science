# import libraries
### Pandas
import pandas as pd
### Regex
import re
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
### GridSearch
from sklearn.model_selection import train_test_split, GridSearchCV

# load messages dataset
messages = pd.read_csv('messages.csv')

# load categories dataset
categories = pd.read_csv('categories.csv')

# merge datasets
df = messages.merge(categories, on='id')

# create a dataframe of the 36 individual category columns
categories = categories['categories'].str.split(';', expand=True)

# select the first row of the categories dataframe
row = categories.loc[0,:]

# use this row to extract a list of new column names for categories.
# one way is to apply a lambda function that takes everything 
# up to the second to last character of each string with slicing
category_colnames = row.apply(lambda col: col[:-2])
print(category_colnames)

# rename the columns of `categories`
categories.columns = category_colnames

# convert columns values
for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].str.slice(-1)
    
    # convert column from string to numeric
    categories[column] = categories[column].astype(str)

# drop the original categories column from `df`
df = df.drop(labels='categories', axis=1)

# concatenate the original dataframe with the new `categories` dataframe
df = df.merge(categories, left_index=True, right_index=True)

# check number of duplicates
df.count()

# drop duplicates
df_nonduplicate = df.drop_duplicates()

# check number of duplicates
df_nonduplicate.count()

engine = create_engine('sqlite:///InsertDatabaseName.db')
df_nonduplicate.to_sql('InsertTableName', engine, index=False)