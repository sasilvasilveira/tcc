import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame

from src.bug import Bug

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer


class Algorithms:
    def __init__(self, df: DataFrame) -> None:
        self.data = df
        self.bug_list = []

    def create_bug_instance_list(self) -> None:
        """
        Each row in the dataframe will be converted to a
        Bug instance and added to the self.bug_list
        """
        for _, row in self.data.iterrows():
            current_bug_overview = row[0]
            current_bug_development_step_identification = row[1]
            current_bug_classification = row[2]
            current_bug_cause = row[3]
            current_bug_description = row[4]

            self.bug_list.append(
                Bug(
                    current_bug_overview,
                    current_bug_description,
                    current_bug_classification,
                    current_bug_development_step_identification,
                    current_bug_cause
                )
            )

    def create_training_and_testing_sets(self):

        atributos = ["overview", "description", "development_step_identification", "root_cause"]

        atributos_data = [
            [getattr(dado, atributo) if atributo != "classification" else (
                getattr(dado.classification, sub_atributo) if sub_atributo != "category" else getattr(
                    dado.classification, sub_atributo
                )
            ) for sub_atributo in ["category", "sub_category", "description"] for atributo in atributos]
            for dado in self.bug_list
        ]
        rotulos_data = [getattr(dado.classification, "category") for dado in self.bug_list]

        x_train, x_test, y_train, y_test = train_test_split(
            atributos_data,
            rotulos_data,
            test_size=0.25,
            random_state=42
        )


        print(f'Size of x_train: {len(x_train)}')
        print(f'Size of y_train: {len(y_train)}')
        print(f'Size of x_test: {len(x_test)}')
        print(f'Size of y_test: {len(y_test)}')

        return (x_train, x_test, y_train, y_test)

    def dummy_fun(self, doc):
        return doc

    def naive_bayes(self):

        x_train, x_test, y_train, y_test = self.create_training_and_testing_sets()

        # Create a TF-IDF vectorizer.
        # Uses TF-IDF vectorization to transform text data into numeric vectors.
        # max_features is to limit the number of features (words) from the dataset for
        # which we want to calculate the TF-IDF scores
        tfidf_vectorizer = TfidfVectorizer(
            analyzer='word',
            tokenizer=self.dummy_fun,
            preprocessor=self.dummy_fun,
            token_pattern=None
        )

        # Transform training and testing data into TF-IDF vectors
        X_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
        X_test_tfidf = tfidf_vectorizer.transform(x_test)

        # Train a classification model (using Naive Bayes)
        classifier = MultinomialNB()
        classifier.fit(X_train_tfidf, y_train)

        # Make predictions on the test set
        y_pred = classifier.predict(X_test_tfidf)

        # Evaluate the model's performance
        print(classification_report(y_test, y_pred))