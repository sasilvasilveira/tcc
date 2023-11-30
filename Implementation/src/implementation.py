import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from pandas import DataFrame

from src.constants import ROOT_CAUSE_CLASSIFICATION
from src.db_clean import CleanDatabase

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


class Implementation:
    def __init__(self) -> None:
        self.data = None
