import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import *
import joblib
import pathlib
import tldextract

class CustomClassifier:
    def __init__(self) -> None:
        self._classifier = BernoulliNB()
        self._vectorizer = TfidfVectorizer(ngram_range=(3, 4), analyzer='char_wb')

    def fit(self, x: pd.Series, y: pd.Series) -> None:
        x_train = [tldextract.extract(item).subdomain + tldextract.extract(item).domain for item in x]
        y_train = y
        X_train_vect = self._vectorizer.fit_transform(x_train)
        self._classifier.fit(X_train_vect, y_train)
    
    def predict(self, x: pd.Series) -> np.ndarray:
        x_test = x.str.split('.').str[0]
        X_test_vect = self._vectorizer.transform(x_test)
        return self._classifier.predict(X_test_vect)
    
    def save(self) -> None:
        pathlib.Path('model').mkdir(exist_ok=True) 
        joblib.dump(self._classifier, 'model/clas.pkl')
        joblib.dump(self._vectorizer, 'model/vect.pkl')

    def load(self) -> None:
        self._classifier = joblib.load('model/clas.pkl')
        self._vectorizer = joblib.load('model/vect.pkl')


def main():
    df = pd.read_csv('train.csv').dropna()
    x_train, y_train = df.domain, df.is_dga

    q = CustomClassifier()
    q.fit(x_train, y_train)
    q.save()

if __name__ == '__main__':
    main()