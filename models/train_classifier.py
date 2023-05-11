import sys
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
import re
from time import time

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, make_scorer, fbeta_score
from sklearn.multioutput import MultiOutputClassifier

import cloudpickle

nltk.download(['punkt', 'wordnet', 'stopwords'])



def load_data(database_filepath):
    engine = create_engine("sqlite:///" + database_filepath)

    df = pd.read_sql('SELECT * FROM disaster_data', engine)

    df.drop(['genre', 'related'], axis = 1, inplace = True)

    #df_sample = df.sample(frac = 0.025, replace = True, random_state = 123)

    return df


def tokenize(text):
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls = re.findall(url_regex, text)
    
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    text = text.lower()

    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    
    text = text.strip()
    
    tokens = word_tokenize(text)
    
    tokens = [t for t in tokens if t not in stopwords.words('english')]
        
    lemmatizer = WordNetLemmatizer()

    clean_tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return clean_tokens



def build_model():

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer= tokenize, max_features= 1000)), 
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state= 123)))
    ])

    parameters = {
        'clf__estimator__min_samples_leaf': [3, 4, 5], 
        'clf__estimator__n_estimators': [ 200, 300, 400]
    }

    ftwo_scorer = make_scorer(fbeta_score, beta = 2, average = 'macro')


    cv = GridSearchCV(pipe, param_grid= parameters, scoring= ftwo_scorer)

    return cv 





def evaluate_model(model, X_test, Y_test):
    predictions = model.predict(X_test)

    cv_results = pd.DataFrame(model.cv_results_)

    cv_results.to_csv('models/cv_results2.csv')
    
    return print(classification_report(y_true= Y_test, 
                                       y_pred= predictions, 
                                       target_names = Y_test.columns))




def save_model(model, model_filepath):
    return cloudpickle.dump(model, open(model_filepath, 'wb'))




def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        df = load_data(database_filepath)

        X_train, X_test, Y_train, Y_test = train_test_split(df['message'], df.drop(['message'], axis = 1), random_state = 123, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        t0 = time()
        model.fit(X_train, Y_train)
        print(f"Model trained. It took {time() - t0:.3f}s")
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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