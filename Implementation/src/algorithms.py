import numpy as np
from pandas import DataFrame

from src.bug import Bug


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, GlobalMaxPooling1D, Dropout
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
            current_bug_overview = row.iloc[0]
            current_bug_development_step_identification = row.iloc[1]
            current_bug_classification = row.iloc[2]
            current_bug_cause = row.iloc[3]
            current_bug_description = row.iloc[4]

            self.bug_list.append(
                Bug(
                    current_bug_overview,
                    current_bug_description,
                    current_bug_classification,
                    current_bug_development_step_identification,
                    current_bug_cause,
                )
            )

    def create_training_and_testing_sets_NB(self):
        features = []
        labels = []

        for bug in self.bug_list:
            features.append(bug.classification.description)
            labels.append(bug.classification.sub_category)

        # Splitting the data between test and training sets
        x_train, x_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )

        print(f"Size of x_train: {len(x_train)}")
        print(f"Size of y_train: {len(y_train)}")
        print(f"Size of x_test: {len(x_test)}")
        print(f"Size of y_test: {len(y_test)}")

        return x_train, x_test, y_train, y_test

    def create_training_and_testing_sets_RNN(self):
        # Mapping strings to numbers
        unique_labels = list(
            set([c.classification.sub_category for c in self.bug_list])
        )
        label_to_index = {label: i for i, label in enumerate(unique_labels)}
        index_to_label = {i: label for label, i in label_to_index.items()}

        x_data = np.array(
            [label_to_index[bug.classification.sub_category] for bug in self.bug_list]
        )
        y_data = to_categorical(x_data, num_classes=len(unique_labels))

        # Splitting the data between test and training sets
        x_train, x_test, y_train, y_test = train_test_split(
            np.expand_dims(x_data, axis=-1), y_data, test_size=0.2, random_state=42
        )

        print(f"Size of x_train: {len(x_train)}")
        print(f"Size of y_train: {len(y_train)}")
        print(f"Size of x_test: {len(x_test)}")
        print(f"Size of y_test: {len(y_test)}")

        return x_train, x_test, y_train, y_test, index_to_label

    def dummy_fun(self, doc):
        return doc

    def naive_bayes(self):
        x_train, x_test, y_train, y_test = self.create_training_and_testing_sets_NB()

        # Vectorizing the features
        vectorizer = CountVectorizer()
        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(x_test)

        # Model training
        model = MultinomialNB()
        model.fit(x_train, y_train)

        # Models testing
        y_pred = model.predict(x_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(conf_matrix)

        # Classification Report
        class_report = classification_report(y_test, y_pred)
        print("Classification Report:")
        print(class_report)

    def redes_neurais_recorrentes(self):
        (
            x_train,
            x_test,
            y_train,
            y_test,
            index_to_label,
        ) = self.create_training_and_testing_sets_RNN()

        # Model definition
        model = Sequential()
        model.add(
            Embedding(input_dim=len(np.unique(x_train)), output_dim=8, input_length=1)
        )
        model.add(LSTM(100))
        model.add(Dense(len(np.unique(x_train)), activation="softmax"))

        # Model Compilation
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        # Model training
        model.fit(x_train, y_train, epochs=12, batch_size=1)

        # Models testing
        prediction = model.predict(x_test)
        print(prediction)

        # Evaluate the model on the test sets
        evaluation = model.evaluate(x_test, y_test)
        print("Loss:", evaluation[0])
        print("Accuracy:", evaluation[1])

        # Get the models predictions on the test set
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        # Converting the indexes back to their original categories
        y_true_labels = [index_to_label[i] for i in y_true_classes]
        y_pred_labels = [index_to_label[i] for i in y_pred_classes]

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
        print("Confusion Matrix:")
        print(conf_matrix)

        # Classification Report
        class_report = classification_report(y_true_labels, y_pred_labels)
        print("Classification Report:")
        print(class_report)
