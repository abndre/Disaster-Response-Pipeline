import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from termcolor import colored, cprint


nltk.download(['wordnet', 'punkt', 'stopwords'])


def load_data(database_filepath):
    """
       Function:
       load data from database
       Args:
       database_filepath: the path of the database
       Return:
       X (DataFrame) : Message features dataframe
       Y (DataFrame) : target dataframe
       category (list of str) : target labels list
       """
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table('disaster_messages_tbl', engine)
    X = df['message']  # Message Column
    Y = df.iloc[:, 4:]  # Classification label

    # listing the columns
    category_names = list(np.array(Y.columns))

    return X, Y, category_names


def tokenize(text):
    """
    Function: split text into words and return the root form of the words
    Args:
      text(str): the message
    Return:
      lemm(list of str): a list of the root form of the message words
    """

    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize text
    words = word_tokenize(text)

    # Remove stop words
    stop = stopwords.words("english")
    words = [t for t in words if t not in stop]

    # Lemmatization
    lemm = [WordNetLemmatizer().lemmatize(w) for w in words]

    return lemm


def build_model():
    """
     Function: build a model for classifing the disaster messages
     Return:
       cv(list of str): classification model
     """

    # Create a pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    # Create Grid search parameters
    parameters = {
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 60, 70]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """
    inputs
        model
        X_test
        y_test
        category_names
    output:
        scores
    """
    y_pred = model.predict(X_test)

    for i, col in enumerate(category_names):
        precision, recall, fscore, support = precision_recall_fscore_support(y_test[col],
                                                                    y_pred[:, i],
                                                                    average='weighted')

        print('\nReport for the column ({}):\n'.format(colored(col, 'red', attrs=['bold', 'underline'])))

        if precision >= 0.75:
            print('Precision: {}'.format(colored(round(precision, 2), 'green')))
        else:
            print('Precision: {}'.format(colored(round(precision, 2), 'yellow')))

        if recall >= 0.75:
            print('Recall: {}'.format(colored(round(recall, 2), 'green')))
        else:
            print('Recall: {}'.format(colored(round(recall, 2), 'yellow')))

        if fscore >= 0.75:
            print('F-score: {}'.format(colored(round(fscore, 2), 'green')))
        else:
            print('F-score: {}'.format(colored(round(fscore, 2), 'yellow')))

def save_model(model, model_filepath):
    """
    Function: Save a pickle file of the model
    Args:
    model: the classification model
    model_filepath (str): the path of pickle file
    """

    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y , category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    print("START")
    main()
    print("END")
