import pandas as pd
from training import load_and_prepare_data, recent_return_rate, encode_columns, scale_data, add_calculated_columns 
from data_ingestion import DataLoader
import logging
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(path):
    try:
        # Load the model using joblib
        return joblib.load(path)
    except Exception as e:
        logging.error(f"Failed to load the model from {path}: {e}")
        raise
    
def load_and_prepare_data(data_loader):
    try:
        train, test = data_loader.load_data()
        train['OrderDate'] = pd.to_datetime(train['OrderDate'])
        train['CustomerBirthDate'] = pd.to_datetime(train['CustomerBirthDate'])
        test['OrderDate'] = pd.to_datetime(test['OrderDate'])
        test['CustomerBirthDate'] = pd.to_datetime(test['CustomerBirthDate'])
    except KeyError as e:
        logging.error(f"Missing essential column in the dataset: {e}")
        raise KeyError(f"Missing essential column in the dataset: {e}")
    except Exception as e:
        logging.error(f"An error occurred while loading and preparing data: {e}")
        raise Exception(f"An error occurred while loading and preparing data: {e}")
    return train, test

file_path = 'credentials/cred.json'

# Open the file and load its contents into a dictionary
with open(file_path, 'r') as file:
    creds = json.load(file)
            
# Initialize DataLoader
data_loader = DataLoader(train_path=creds['training_data_path'], test_path=creds['testing_data_path'])

# Load and prepare data
train, test = load_and_prepare_data(data_loader)

# Load additional datasets
repeat_returns = pd.read_csv(creds['repeat_returns_path'])
test = pd.merge(test, repeat_returns, on='CustomerID', how='left')
test['RepeatReturnFlag'] = test['RepeatReturnFlag'].fillna(0).astype(int)

product_returns = pd.read_csv(creds['product_returns_path'])
test = pd.merge(test, product_returns, on='ProductDepartment', how='left')
test['ProductReturnRate'] = test['ProductReturnRate'].fillna(0)


# Add calculated columns
test = add_calculated_columns(test)

# Add a marker column to distinguish between train and test in the concatenated DataFrame
train['dataset'] = 'train'
test['dataset'] = 'test'

# Concatenate train and test for the purpose of calculating RecentReturnRate
recent_return_df = pd.concat([train, test])

# Sort by CustomerID and OrderDate
sorted_recent_return_df = recent_return_df.sort_values(by=['CustomerID', 'OrderDate'])

# Add calculated columns
sorted_recent_return_df = add_calculated_columns(sorted_recent_return_df)

# Filter out the train entries and keep only the test entries with updated RecentReturnRate
test = sorted_recent_return_df[sorted_recent_return_df['dataset'] == 'test'].drop(columns=['dataset'])

test = test.fillna(0)

# Define required columns
with open(r'config/features.yaml', 'r') as file:
    config = yaml.safe_load(file)
    
cols_reqd = config['required_columns']
cat_columns = config['categorical_columns']
continuous_columns = config['continuous_columns']

test_reqd = test[cols_reqd]

# One Hot Encoding the required columns
encoded_data = encode_columns(test_reqd, cat_columns, method='onehot')
logging.info("One-hot encoding complete.")

# Scaling specified columns
final_train = scale_data(encoded_data, continuous_columns)

# Assuming the data is already encoded and split into X and y
X_test = encoded_data.drop(['Returned'], axis=1)

# Load the model
loaded_model = load_model(creds['model_save_path'])

# Load the columns used for training
train_X = pd.read_csv(creds['preprocessed_X_path'])
X_train_columns = train_X.columns

# Drop 'Unnamed: 0' if it exists in X_train_columns
if 'Unnamed: 0' in X_train_columns:
    X_train_columns = X_train_columns.drop('Unnamed: 0')

# Ensure the test data has the same columns as the training data, in the same order
X_train_columns = pd.Index([col for col in X_train_columns if col in X_test.columns and X_train_columns.duplicated().sum() == 0])
X_test = X_test.loc[:, ~X_test.columns.duplicated()]  # Remove duplicate columns in X_test
X_test = X_test.reindex(columns=X_train_columns).reset_index(drop=True)

# Assuming the test data is already prepared and encoded in X_test
predictions = loaded_model.predict(X_test)

# Adding predictions to the test data DataFrame
X_test['Returned'] = predictions 
X_test.to_csv(creds['predictions_csv_path'], index=False)
logging.info("Predictions made and saved successfully.")

features = X_test.columns
if hasattr(loaded_model, 'feature_importances_'):
    importances = loaded_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    # Plot the feature importances of the model
    plt.figure(figsize=(25, 10))
    plt.title(f"Feature importances")
    sns.barplot(y=[features[i] for i in indices], x=importances[indices], orient='h')
    plt.show()
