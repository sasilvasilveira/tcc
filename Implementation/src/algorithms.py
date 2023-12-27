import numpy as np
from pandas import DataFrame

from src.bug import Bug

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, GlobalMaxPooling1D, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint


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

    def create_training_and_testing_sets_NB(self):

        attributes = ["overview", "description", "development_step_identification", "root_cause"]

        data_attributes = [
            [getattr(data, attribute) if attribute != "classification" else (
                getattr(data.classification, sub_atribute) if sub_atribute != "sub_category" else getattr(
                    data.classification, sub_atribute
                )
            ) for sub_atribute in ["category", "sub_category", "description"] for attribute in attributes]
            for data in self.bug_list
        ]
        data_labels = [getattr(data.classification, "sub_category") for data in self.bug_list]

        x_train, x_test, y_train, y_test = train_test_split(
            data_attributes,
            data_labels,
            test_size=0.3,
            random_state=42
        )


        print(f'Size of x_train: {len(x_train)}')
        print(f'Size of y_train: {len(y_train)}')
        print(f'Size of x_test: {len(x_test)}')
        print(f'Size of y_test: {len(y_test)}')

        return (x_train, x_test, y_train, y_test)
    
    def create_training_and_testing_sets_RNN(self):
        # Mapping strings to numbers
        unique_labels = list(set([c.classification.category for c in self.bug_list]))
        label_to_index = {label: i for i, label in enumerate(unique_labels)}

        x_data = np.array([label_to_index[bug.classification.category] for bug in self.bug_list])
        y_data = to_categorical(x_data, num_classes=len(unique_labels))

        return np.expand_dims(x_data, axis=-1), y_data

    def dummy_fun(self, doc):
        return doc

    def naive_bayes(self):

        x_train, x_test, y_train, y_test = self.create_training_and_testing_sets_NB()

        # Create a TF-IDF vectorizer.
        # Uses TF-IDF vectorization to transform text data into numeric vectors
        tfidf_vectorizer = TfidfVectorizer(
            analyzer='word',
            tokenizer=self.dummy_fun,
            preprocessor=self.dummy_fun,
            token_pattern=None
        )

        # Transform training and testing data into TF-IDF vectors
        x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
        x_test_tfidf = tfidf_vectorizer.transform(x_test)

        # Train a classification model (using Naive Bayes)
        classifier = MultinomialNB()
        classifier.fit(x_train_tfidf, y_train)

        # Make predictions on the test set
        y_pred = classifier.predict(x_test_tfidf)

        # Evaluate the model's performance
        results_classification_report = classification_report(y_test, y_pred)
        print(results_classification_report)

    def redes_neurais_recorrentes(self):
        
        x_train, y_train = self.create_training_and_testing_sets_RNN()

        # Model definition
        model = Sequential()
        model.add(Embedding(input_dim=len(np.unique(x_train)), output_dim=8, input_length=1))
        model.add(LSTM(100))
        model.add(Dense(len(np.unique(x_train)), activation='softmax'))

        # Model Compilation
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Model training
        model.fit(x_train, y_train, epochs=10, batch_size=1)

        # Models testing
        prediction = model.predict(y_train)
        print(prediction)
