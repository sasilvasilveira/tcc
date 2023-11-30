from src.implementation import Implementation
from src.db_clean import CleanDatabase
from src.constants import BUG_CATEGORY_COLUMN_NAME, COLUMNS_TO_REMOVE

def main():
    clean_database = CleanDatabase(
        'src/amostra-bugs.csv',
        BUG_CATEGORY_COLUMN_NAME,
        COLUMNS_TO_REMOVE
    )
    clean_database.clean_database_process()
    print(clean_database.data)


if __name__ == '__main__':
    main()