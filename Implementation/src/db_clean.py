import pandas as pd
import string
import nltk
nltk.download('stopwords') # to download stopwords corpus

from nltk.corpus import stopwords
from pandas import DataFrame
from src.constants import COLUMNS_WITH_SAME_WORDS_AS_REMAINING_COLUMNS


class CleanDatabase:
    def __init__(
        self,
        csv_file_path: str,
        bug_category_column_name: str,
        columns_to_remove: list
    ) -> None:

        self.data = pd.read_csv(csv_file_path)
        self.columns_to_remove = columns_to_remove
        self.bug_category_column_name = bug_category_column_name
        self.cause_column_name = ""
        self.remaining_columns = []

    def drop_unwanted_columns(self):
        """
        Method to remove all columns found at self.columns_to_remove
        from the self.dataframe
        """
        # Columns with similar words to be removed
        for column_name in self.columns_to_remove:
            self.data = self.data.drop(
                self.data.filter(regex=column_name).columns,
                axis=1
            )

        # Specific columns to be dropped
        for specific_name in COLUMNS_WITH_SAME_WORDS_AS_REMAINING_COLUMNS:
            self.data = self.data.drop(
                columns=[specific_name],
                axis=1
            )

    def to_lower_case_all_fields(self):
        """
        Method to convert all field values to lower case
        """
        for column in self.data.columns:
            self.data[column] = self.data[column].str.lower()

    def remove_stop_words(self):
        """
        Method to remove all stop words from the database
        """
        for column in self.data.columns:
            self.data[column] = self.data[column].apply(
                lambda x: ' '.join(
                    [word for word in x.split() if word not in (
                        stopwords.words('portuguese')
                    )]
                )
            )

    def remove_short_words(self):
        """
        Method to remove all words shorter than 2 chars from
        the database
        """
        for column in self.data.columns:
            self.data[column] = self.data[column].apply(
                lambda x: ' '.join(
                    word for word in x.split() if len(word)>2
                )
            )

    def set_columns_classification(self):
        """
        Method to set all columns according to the class's
        classification
        """
        for column in self.data.columns:
            if ('Causa' not in column) and (
                self.bug_category_column_name not in column
            ):
                self.remaining_columns.append(column)
            elif 'Causa' in column:
                self.cause_column_name = column
        self.remaining_columns.sort()

    def clean_database_process(self) -> DataFrame:
        """
        Method to clean the dataframe according to project rules

        :return: the cleaned given dataframe
        :rtype: DataFrame
        """

        # Removing all columns with NaN values
        self.data = self.data.dropna(axis = 1, how = 'all')
        # Removing all rows with NaN values at bug category column
        self.data = self.data[self.data[self.bug_category_column_name].notna()]

        self.drop_unwanted_columns()
        self.to_lower_case_all_fields()
        self.remove_stop_words()
        self.remove_short_words()
        self.set_columns_classification()

        print('Remaining columns:\n')
        for column in self.remaining_columns:
            print(column)
        print('\n')
        print(f'Cause column:\n{self.cause_column_name}')
        print('\n')
        print(f'Classification column:\n{self.bug_category_column_name}')

        return self.data
