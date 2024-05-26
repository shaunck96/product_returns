import pandas as pd
import logging

# Configure logging to display the time, level of severity, and message of logs.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    """
    A class for loading and checking data quality of datasets.
    
    Attributes:
        train_path (str): Path to the training dataset CSV file.
        test_path (str): Path to the testing dataset CSV file.
        train (DataFrame): DataFrame containing the training data.
        test (DataFrame): DataFrame containing the testing data.
    """

    def __init__(self, train_path='train.csv', test_path='test.csv'):
        """
        The constructor for DataLoader class.

        Parameters:
            train_path (str): Path to the training dataset CSV file.
            test_path (str): Path to the testing dataset CSV file.
        """
        try:
            self.train = pd.read_csv(train_path)
            self.test = pd.read_csv(test_path)
        except FileNotFoundError as e:
            logging.critical(f"File not found: {e}")
            raise
        except pd.errors.EmptyDataError as e:
            logging.critical(f"Empty data error: {e}")
            raise
        except pd.errors.ParserError as e:
            logging.critical(f"Parsing error: {e}")
            raise

    def check_data_quality(self):
        """
        Executes all data quality checks on the datasets.
        """
        self.check_dates()
        self.check_strings()
        self.check_floats()

    def check_dates(self):
        """
        Check for date-related errors such as incorrect formats and future dates.
        """
        date_columns = ['CustomerBirthDate', 'OrderDate']
        for column in date_columns:
            for df_name, df in [('train', self.train), ('test', self.test)]:
                if column in df.columns:
                    try:
                        dates = pd.to_datetime(df[column], errors='coerce')
                        if not dates.notnull().all():
                            logging.warning(f"Date format error in {column} in {df_name} dataset")
                        if dates.max() > pd.Timestamp.today():
                            logging.warning(f"Future date found in {column} in {df_name} dataset")
                    except Exception as e:
                        logging.error(f"Error checking dates in column {column} in {df_name} dataset: {e}")

    def check_strings(self):
        """
        Check for issues in string columns like missing values and duplicate IDs.
        """
        string_columns = ['ID', 'OrderID', 'CustomerID', 'ProductDepartment']
        for column in string_columns:
            for df_name, df in [('train', self.train), ('test', self.test)]:
                if column in df.columns:
                    if df[column].isnull().any():
                        logging.warning(f"Missing values found in {column} in {df_name} dataset")
                    if column in ['ID', 'OrderID', 'CustomerID'] and df[column].duplicated().any():
                        logging.warning(f"Duplicate IDs found in {column} in {df_name} dataset")

    def check_floats(self):
        """
        Check for issues in float columns like negative values and unreasonable percentages.
        """
        float_columns = ['ProductCost', 'DiscountPct', 'PurchasePrice']
        for column in float_columns:
            for df_name, df in [('train', self.train), ('test', self.test)]:
                if column in df.columns:
                    if (df[column] < 0).any():
                        logging.warning(f"Negative values found in {column} in {df_name} dataset")
                    if column == 'DiscountPct' and (df[column] > 100).any():
                        logging.warning(f"Discount percentage over 100 in {column} in {df_name} dataset")

    def convert_column_types(self):
        """
        Convert columns to their appropriate data types.
        """
        try:
            for df_name, df in [('train', self.train), ('test', self.test)]:
                df[['CustomerBirthDate', 'OrderDate']] = df[['CustomerBirthDate', 'OrderDate']].apply(pd.to_datetime)
                df[['ID', 'OrderID', 'CustomerID', 'ProductDepartment']] = df[['ID', 'OrderID', 'CustomerID', 'ProductDepartment']].astype(str)
                df[['ProductCost', 'DiscountPct', 'PurchasePrice']] = df[['ProductCost', 'DiscountPct', 'PurchasePrice']].astype(float)
                if 'Returned' in df.columns:
                    df['Returned'] = df['Returned'].astype(int)
        except Exception as e:
            logging.error(f"Error converting column types: {e}")

    def load_data(self):
        """
        Returns the training and testing datasets.

        Returns:
            tuple: A tuple containing the training and testing DataFrames.
        """
        return self.train, self.test
