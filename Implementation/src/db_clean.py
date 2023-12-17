import pandas as pd
import nltk
nltk.download('stopwords') # to download stopwords corpus

from nltk.corpus import stopwords
from pandas import DataFrame
from src.classification import Classification
from src.constants import COLUMNS_WITH_SAME_WORDS_AS_REMAINING_COLUMNS, ROOT_CAUSE_CLASSIFICATION


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

    def classify_bug_category(self) -> None:
        """
        Replacing the bug categories entries for the ones found
        at the documentation (ROOT_CAUSE_CLASSIFICATION constant)
        """
        for item in self.data[self.bug_category_column_name]:
            for key, value in ROOT_CAUSE_CLASSIFICATION.items():
                for sub_values in value:
                    sub_key, description = sub_values.split(":")
                    if item.lower() in sub_values.lower():
                        bug_category_classification = Classification(
                            key, sub_key.strip(), description.strip() 
                        )

                        self.data[self.bug_category_column_name] = (
                            self.data[self.bug_category_column_name].replace(
                                item, bug_category_classification
                            )
                        )

    def drop_unwanted_columns(self) -> None:
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
    
    def drop_unwanted_rows(self) -> None:
        """
        Method to remove all rows from the self.dataframe where
        its values are different from Classification class instance
        at self.bug_category_column_name
        """        
        self.data = self.data.loc[self.data[
            self.bug_category_column_name
        ].apply(lambda x: isinstance(x, Classification))]

    def to_lower_case_all_fields(self):
        """
        Method to convert all field values to lower case
        """
        for column in self.data.columns:
            for item in self.data[column]:
                if isinstance(item, str):
                    self.data[column] = self.data[column].replace(item, item.lower())

    def remove_stop_words(self) -> None:
        """
        Method to remove all stop words from the database
        """

        #  Verified that this stop word is considered important
        #  for the problem's understanding
        stop_words = list(stopwords.words('portuguese'))
        stop_words.remove('não')

        for column in self.data.columns:
            if column != self.bug_category_column_name:
                self.data[column] = self.data[column].apply(
                    lambda x: ' '.join(
                        word for word in x.split() if
                        word not in stop_words
                    )
                )

    def remove_short_words(self) -> None:
        """
        Method to remove all words shorter than 2 chars from
        the database
        """
        for column in self.data.columns:
            if column != self.bug_category_column_name:
                self.data[column] = self.data[column].apply(
                    lambda x: ' '.join(
                        word for word in x.split() if len(word)>2
                    )
                )

    def set_columns_classification(self) -> None:
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

        self.classify_bug_category()
        self.drop_unwanted_rows()
        self.drop_unwanted_columns()
        self.to_lower_case_all_fields()
        self.remove_stop_words()
        self.remove_short_words()
        self.set_columns_classification()
