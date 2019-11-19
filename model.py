import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from datetime import datetime


class Model(object):

    def __init__(self, review, y):
        self.review = review
        self.y = y

    def run_model(self):
        tfidfconverter = TfidfVectorizer(max_features=1500, max_df=0.7, ngram_range=(1, 2))
        X = tfidfconverter.fit_transform(self.review).toarray()
        y = self.y

        logging.info('text processing finished!')
        dt = datetime.now()
        classifier = RandomForestClassifier(n_estimators=600, max_depth=10, random_state=dt.second)
        cv = StratifiedKFold(n_splits=5, random_state=dt.second, shuffle=True)

        for (train, test), i in zip(cv.split(X, y), range(5)):
            classifier.fit(X[train], y[train])
            logging.info('model trained!')
            y_pred = classifier.predict(X[test])

            print(classification_report(self.y[test], y_pred))
            print(accuracy_score(self.y[test], y_pred))

