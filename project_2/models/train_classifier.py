import sys
### Pandas
import pandas as pd
### NumPy
import numpy as np
### SQL Alchemy
import sqlalchemy
### Import nltk
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
### Stemming & Lemming
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
### Pipeline
from sklearn.pipeline import Pipeline, FeatureUnion
### Pipeline transformation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
### GridSearch
from sklearn.model_selection import train_test_split, GridSearchCV
### Output Classifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
### Confusion Matrix
from sklearn.metrics import f1_score, classification_report, accuracy_score, confusion_matrix
### Pickle
import pickle
### Time
import time


def load_data(database_filepath):
    engine = sqlalchemy.create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('messages', engine)
    
    categories = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = categories.columns
    
    X = df.message.values
    Y = df[category_names]
    
    return X, Y, category_names


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    knn = KNeighborsClassifier(n_neighbors=3)

    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
        ])),
        ('clf', MultiOutputClassifier(knn))
    ])
    print(pipeline.get_params())
    
    parameters = {
        'features__text_pipeline__tfidf__smooth_idf': [True, False],
        'features__text_pipeline__tfidf__sublinear_tf': [False, True],
        # 'clf__estimator__leaf_size': [30, 50, 100],
        # 'clf__estimator__p':[2, 3]
    }
    
    gridsearch = GridSearchCV(pipeline, param_grid=parameters, refit=True)

    return gridsearch


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    labels = np.unique(y_pred)
    # confusion_mat = confusion_matrix(Y_test, y_pred, labels=category_names)
    
    accuracy = (y_pred == Y_test).mean()
    print(accuracy)
    print("\nBest Parameters:", model.best_params_)

    print("Labels:")
    print(labels)
    # Crash due to "ValueError: multiclass-multioutput is not supported":
    # print(f1_score(Y_test, y_pred))
    # print(classification_report(Y_test, y_pred, labels=labels))


def save_model(model, model_filepath):
    save_file = open(model_filepath, "wb")
    pickle.dump(model, save_file)
    save_file.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        
        start = time.time()
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)
        end = time.time()
        print('Done in {} sec(s)'.format(end-start))
        
        print('Building model...')
        start = time.time()
        model = build_model()
        end = time.time()
        print('Done in {} sec(s)'.format(end-start))
        
        print('Training model...')
        start = time.time()
        model.fit(X_train, Y_train)
        print("\nBest Parameters:", model.best_params_)
        end = time.time()
        print('Done in {} sec(s)'.format(end-start))
        
        print('Evaluating model...')
        start = time.time()
        evaluate_model(model, X_test, Y_test, category_names)
        end = time.time()
        print('Done in {} sec(s)'.format(end-start))

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
