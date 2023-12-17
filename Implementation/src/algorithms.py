import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from pandas import DataFrame

from src.constants import ROOT_CAUSE_CLASSIFICATION
from src.db_clean import CleanDatabase
from src.bug import Bug

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


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

            self.bug_list.append(Bug(
                current_bug_overview,
                current_bug_description,
                current_bug_classification,
                current_bug_development_step_identification,
                current_bug_cause
            ))
