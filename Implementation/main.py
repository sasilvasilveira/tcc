from src.db_clean import CleanDatabase
from src.algorithms import Algorithms
from src.constants import BUG_CATEGORY_COLUMN_NAME, COLUMNS_TO_REMOVE

def main():
    clean_database = CleanDatabase(
        'src/amostra-bugs.csv',
        BUG_CATEGORY_COLUMN_NAME,
        COLUMNS_TO_REMOVE
    )
    clean_database.clean_database_process()

    algorithms = Algorithms(clean_database.data)
    algorithms.create_bug_instance_list()

    algorithms.naive_bayes()

if __name__ == '__main__':
    main()