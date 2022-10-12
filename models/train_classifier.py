""" Build and train a model to classify disaster messages"""
# import libraries
import sys
import re
import pickle
import pandas as pd
from sqlalchemy import create_engine
# Steven Bird, Ewan Klein, and Edward Loper (2009). Natural Language Processing with Python.
# O'Reilly Media Inc.  http://nltk.org/book
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

# library for creating a cutomize transformer
from sklearn.base import BaseEstimator, TransformerMixin
# sentence tozenize
from nltk import sent_tokenize
# tokenizing into word and tag of a speech
from nltk import pos_tag


def load_data(database_filepath):
    """load the data from the database

    Args:
        database_filepath (str): the path to the database

    Returns:
        X, Y, Y.columns: the features, labels and category names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(engine.table_names()[0], engine)
    X = df.message
    Y = df[df.columns[4:]]

    return X, Y, Y.columns


def tokenize(text):
    """tokenize the text

    Args:
        text (text): the text to be tokenized

    Returns:
        clean_text: the tokenized text
    """

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # find all the urls
    find_url = re.findall(url_regex, text)

    # replace the urls with 'urlplaceholder'
    for url in find_url:
        text = text.replace(url, 'urlplaceholder')

    # tokenize the text
    words = word_tokenize(text)
    # lemmatize the words
    # make all the characters lower and remove the punctuations and whitespaces
    clean_text = [WordNetLemmatizer().lemmatize(word).lower().strip() for word in words]

    return clean_text

# Implement the StartingVerbExtractor class
class StartingVerbExtrator(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        """check if the sentence starts with a verb

        Args:
            text (str): the text to be checked

        Returns:
            bool: True if the sentence starts with a verb, False otherwise
        """

        #tokernize the sentence
        tokenize_sentence = sent_tokenize(text)

        # tokenize each sentence into words and tag
        for sentence in tokenize_sentence:
            pos_tags = pos_tag(tokenize(sentence))
            # get the first word and part of speech tag by indexing the pos_tag
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        """fit the model

        Args:
            X (array): the features
            y (array, optional): the labels. Defaults to None.

        Returns:
           self: the fitted model
        """
        return self

    def transform(self, X):
        """transform the data

        Args:
            X (array): the features

        Returns:
            DataFrame: the transformed data
        """
        # apply the custom starting_verb function to all the X values
        X_tagged = pd.Series(X).apply(self.starting_verb)
        # make a dataframe of the result

        return pd.DataFrame(X_tagged)


def build_model():
    """Build the model pipeline

    Returns:
        ml_pipeline: the model pipeline
    """

    ml_pipeline = Pipeline([
            ('features', FeatureUnion([
                ('text_pipeline', Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer())
                ])),
                ('starting_verb', StartingVerbExtrator())
            ])),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])

    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__n_estimators': [50, 100]
        }

    # define a gridsearvh object
    grid_pipeline = GridSearchCV(ml_pipeline, param_grid=parameters, n_jobs=-1)

    return grid_pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """evaluate the model

    Args:
        model: the model to be evaluated
        X_test (DataFrame): the test features
        Y_test (DataFrame): the test labels
        category_names (list): the category names
    """

    # predict the labels
    y_pred = model.best_estimator_.predict(X_test)


    # print the classification report
    for i in range(0, len(Y_test.columns)):
        print(f"in {category_names[i]} the report is \
            {classification_report(Y_test.iloc[:, i].values, y_pred[:, i])}")



def save_model(model, model_filepath):
    """save the model

    Args:
        model: the model to be saved
        model_filepath (str): the path to save the model
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model.best_estimator_, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
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
    main()