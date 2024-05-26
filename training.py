import pandas as pd
import logging
from data_ingestion import DataLoader
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from category_encoders import BinaryEncoder, TargetEncoder, HashingEncoder
import numpy as np
import pickle
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import brier_score_loss
import os
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import joblib
import json
import shutil
import yaml
import importlib
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_folder_writable(folder_path):
    try:
        test_file = os.path.join(folder_path, 'temp_file.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        logging.info(f"Folder '{folder_path}' is writable.")
    except Exception as e:
        logging.error(f"Folder '{folder_path}' is not writable. Error: {e}")
        raise
    
def get_season(date):
    month = date.month
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    else:
        return 'Winter'

def calculate_age(birthdate, orderdate):
    return orderdate.year - birthdate.year - ((orderdate.month, orderdate.day) < (birthdate.month, birthdate.day))

def recent_return_rate(x):
    return x.rolling(window=10, min_periods=1).mean()

def holidays():
    cal = USFederalHolidayCalendar()
    return cal.holidays(start='2017-01-01', end='2024-12-31')

def load_and_prepare_data(data_loader):
    try:
        train, _ = data_loader.load_data()
        train['OrderDate'] = pd.to_datetime(train['OrderDate'])
        train['CustomerBirthDate'] = pd.to_datetime(train['CustomerBirthDate'])
    except KeyError as e:
        logging.error(f"Missing essential column in the dataset: {e}")
        raise KeyError(f"Missing essential column in the dataset: {e}")
    except Exception as e:
        logging.error(f"An error occurred while loading and preparing data: {e}")
        raise Exception(f"An error occurred while loading and preparing data: {e}")
    return train

def add_calculated_columns(df):
    try:
        df['msrp'] = df['PurchasePrice'] * (1 - df['DiscountPct'])
        repeat_returns = df.groupby('CustomerID')['Returned'].sum()
        df['RepeatReturnFlag'] = df['CustomerID'].map(repeat_returns > 1).astype(int)
        multi_item_orders = df.groupby('OrderID').size()
        df['MultiItemOrder'] = df['OrderID'].map(multi_item_orders > 1).astype(int)
        df['Season'] = df['OrderDate'].apply(get_season)
        df['CustomerAge'] = df.apply(lambda row: calculate_age(row['CustomerBirthDate'], row['OrderDate']), axis=1)
        df['Holiday'] = df['OrderDate'].isin(holidays()).astype(int)
        df['DaysSinceFirstOrder'] = (df['OrderDate'] - df.groupby('CustomerID')['OrderDate'].transform('min')).dt.days
        df['CustomerLifetimeValue'] = df.groupby('CustomerID')['PurchasePrice'].transform('sum')
        df['OrderFrequency'] = df.groupby('CustomerID')['CustomerID'].transform('size')
        product_returns = df.groupby('ProductDepartment')['Returned'].mean()
        df['ProductReturnRate'] = df['ProductDepartment'].map(product_returns)
        df['DayOfWeek'] = df['OrderDate'].dt.day_name()
        sorted_train = df.sort_values(by=['CustomerID', 'OrderDate'])
        df['RecentReturnRate'] = sorted_train.groupby('CustomerID', group_keys=False)['Returned'].apply(recent_return_rate)
        df['PriceSensitivity'] = df['DiscountPct'] / df['msrp']
        df['DaysBetweenOrders'] = df.groupby('CustomerID')['OrderDate'].diff().dt.days.fillna(0)
        df['AvgDaysBetweenOrders'] = df.groupby('CustomerID')['DaysBetweenOrders'].transform('mean')
    except KeyError as e:
        logging.error(f"Missing key data required for calculations: {e}")
        raise KeyError(f"Missing key data required for calculations: {e}")
    except Exception as e:
        logging.error(f"An error occurred while adding calculated columns: {e}")
        raise Exception(f"An error occurred while adding calculated columns: {e}")
    return df

def encode_columns(df: pd.DataFrame, columns: list, method: str = 'onehot', target: str = None) -> pd.DataFrame:
    if method == 'target' and target is None:
        raise ValueError("Target column must be specified for target encoding.")

    original_columns = df.columns.difference(columns)
    df_original = df[original_columns].reset_index(drop=True)
    
    encoder = None
    try:
        if method == 'label':
            encoder = LabelEncoder()
            df[columns] = df[columns].apply(lambda col: encoder.fit_transform(col) if col.name in columns else col)
        elif method == 'onehot':
            encoder_path = r"models/.pkl"
            ct = ColumnTransformer(
                [('onehot', OneHotEncoder(sparse_output=False), columns)],
                remainder='drop'
            )
            ct.fit(df)
            with open(encoder_path, 'wb') as file:
                pickle.dump(ct, file)
                
            df_transformed = pd.DataFrame(ct.transform(df), columns=ct.get_feature_names_out())
            df_transformed.reset_index(drop=True, inplace=True)
            df_original = df.drop(columns, axis=1)
            df_original.reset_index(drop=True, inplace=True)
            df = pd.concat([df_transformed, df_original], axis=1)
        elif method == 'binary':
            encoder = BinaryEncoder(cols=columns)
            df = encoder.fit_transform(df)
        elif method == 'frequency':
            df[columns] = df[columns].apply(lambda col: df[col].map(df[col].value_counts(normalize=True)) if col.name in columns else col)
        elif method == 'target':
            encoder = TargetEncoder(cols=columns)
            df[columns] = encoder.fit_transform(df[columns], df[target])
        elif method == 'hash':
            encoder = HashingEncoder(cols=columns, n_components=8)
            df = encoder.fit_transform(df)
        elif method == 'ordinal':
            encoder = OrdinalEncoder()
            df[columns] = encoder.fit_transform(df[columns])
        else:
            raise ValueError("Unsupported encoding method. Valid methods include 'label', 'onehot', 'binary', 'frequency', 'target', 'hash', 'ordinal'.")
    except Exception as e:
        logging.error(f"An error occurred during encoding: {e}")
        raise
    
    df.reset_index(drop=True, inplace=True)
    return pd.concat([df, df_original], axis=1)

def scale_data(encoded_data: pd.DataFrame, columns: list) -> pd.DataFrame:
    scaler_path=r'models/scaler.pkl'
    try:
        # Fit a new scaler
        scaler = StandardScaler()
        scaler.fit(encoded_data[columns])
        with open(scaler_path, 'wb') as file:
            pickle.dump(scaler, file)
        logging.info("Fitted and saved new scaler.")

        # Check if columns to scale are in the DataFrame
        if not set(columns).issubset(encoded_data.columns):
            missing_cols = set(columns) - set(encoded_data.columns)
            logging.error(f"Missing columns in DataFrame that need scaling: {missing_cols}")
            raise ValueError(f"Missing columns in DataFrame that need scaling: {missing_cols}")

        # Scale the specified columns directly
        encoded_data[columns] = scaler.transform(encoded_data[columns])

        encoded_data = encoded_data.loc[:,~encoded_data.columns.duplicated()].copy()
        logging.info("Scaling complete.")
        return encoded_data

    except Exception as e:
        logging.error(f"An error occurred in the feature engineering process: {e}")
        raise

def save_metrics(metrics, folder_path, file_name):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, file_name)
    metrics.to_csv(file_path, index=False)
    logging.info(f"Metrics saved to {file_path}")

# Function to load models from the config file
def load_models(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
        
    models = {}
    for model_name, model_info in config['models'].items():
        module = importlib.import_module(model_info['module'])
        model_class = getattr(module, model_info['class'])
        parameters = model_info.get('parameters', {})
        models[model_name] = model_class(**parameters)
    
    return models

def stratified_training(X, y):
    # Load models from the YAML configuration file
    models = load_models('config/base_models.yaml')
    # Define the number of splits for StratifiedKFold
    n_splits = 8
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=50)

    # Initialize dictionary to store the average scores for each model
    avg_results = {}

    # Define custom scorer if necessary (example for ROC AUC)
    roc_auc_scorer = make_scorer(roc_auc_score, needs_proba=True)

    # Evaluate models using cross-validation and collect average scores
    for name, model in models.items():
        # Collect scores for various metrics
        roc_auc_scores = cross_val_score(model, X, y, scoring=roc_auc_scorer, cv=skf)
        accuracy_scores = cross_val_score(model, X, y, scoring='accuracy', cv=skf)
        precision_scores = cross_val_score(model, X, y, scoring='precision', cv=skf)
        recall_scores = cross_val_score(model, X, y, scoring='recall', cv=skf)
        f1_scores = cross_val_score(model, X, y, scoring='f1', cv=skf)
        brier_scores = cross_val_score(model, X, y, scoring=make_scorer(brier_score_loss, needs_proba=True), cv=skf)

        # Calculate mean scores for each metric
        mean_scores = {
            'Model': name,
            'ROC AUC': np.mean(roc_auc_scores),
            'Accuracy': np.mean(accuracy_scores),
            'Precision': np.mean(precision_scores),
            'Recall': np.mean(recall_scores),
            'F1 Score': np.mean(f1_scores),
            'Brier Score': np.mean(brier_scores)
        }

        avg_results[name] = mean_scores
        print(f"Average Metrics for {name} over {n_splits}-fold CV: {mean_scores}")

    # Find the best model based on average ROC AUC
    best_model_name = max(avg_results, key=lambda x: avg_results[x]['ROC AUC'])
    best_model = models[best_model_name]
    print(f"Best Model Based on Average ROC AUC: {best_model_name}")
    metrics_df = pd.DataFrame(avg_results).transpose()
    save_metrics(metrics_df, "metrics", 'stratified_training_metrics.csv')
    return best_model, best_model_name

# Function to process the hyperparameter configuration
def process_hyperparameters(config):
    hyperparameters = {}
    for model, params in config.items():
        hyperparameters[model] = {}
        for param, details in params.items():
            if details["type"] == "randint":
                hyperparameters[model][param] = randint(details["low"], details["high"]).rvs()
            elif details["type"] == "uniform":
                hyperparameters[model][param] = uniform(details["low"], details["high"] - details["low"]).rvs()
            elif details["type"] == "choice":
                hyperparameters[model][param] = random.choice(details["values"])
    return hyperparameters
    
def tune_and_save_model(X, y, save_path):
    with open(r'config/hyperparameter_tuning_cfg.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Process the hyperparameters
    hyperparameters = process_hyperparameters(config)
    print(hyperparameters)
    param_distributions = process_hyperparameters(config)
    model, name = stratified_training(X, y)
    best_auc = -1
    scoring = {
        'roc_auc': 'roc_auc',  # Use the predefined string which is equivalent to make_scorer(roc_auc_score, needs_proba=True)
        'brier_score': make_scorer(brier_score_loss, needs_proba=True, greater_is_better=False)  # Brier score setup
    }
    try:
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions[name],
            n_iter=50,  # Adjust n_iter to balance between search space and time
            scoring=scoring,
            cv=5,
            random_state=42,
            n_jobs=-1,
            refit='roc_auc'  # Choose a primary metric for refitting
        )
        random_search.fit(X, y)
        logging.info(f"Best parameters for {name}: {random_search.best_params_}")
        logging.info(f"Best ROC AUC score for {name}: {random_search.best_score_}")

        best_auc = random_search.best_score_
        best_model = model.set_params(**random_search.best_params_) if random_search.best_estimator_ else None
        best_model_name = name
        best_model_metrics = pd.DataFrame({
            'Model': [best_model_name],
            'ROC AUC': [best_auc],
            'Parameters': [random_search.best_params_]
        })
        save_metrics(best_model_metrics, "metrics", 'best_model_metrics.csv')
        best_model.fit(X, y) if best_model else logging.warning("No model was fitted.")
    except Exception as e:
        logging.error(f"Error during model tuning for {name}: {e}")

    # Save the best model
    if best_model is not None:
        try:
            ensure_folder_writable(os.path.dirname(save_path))  # Ensure the folder is writable
            os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure the directory exists
            joblib.dump(best_model, save_path)
            logging.info(f"Best overall model {best_model_name} with AUC {best_auc} saved successfully at {save_path}")
        except Exception as e:
            logging.error(f"Error during model tuning for {name}: {e}")


if __name__ == "__main__":
    try:
        # Path to the JSON file
        file_path = 'credentials/cred.json'

        # Open the file and load its contents into a dictionary
        with open(file_path, 'r') as file:
            creds = json.load(file)

        # Instantiate DataLoader, load and preprocess data, and perform feature engineering
        data_loader = DataLoader(train_path=creds['training_data_path'], test_path=creds['testing_data_path'])
        train = load_and_prepare_data(data_loader)
        train = add_calculated_columns(train)

        repeat_returns = train[['CustomerID', 'RepeatReturnFlag']].drop_duplicates()
        product_returns = train[['ProductDepartment', 'ProductReturnRate']].drop_duplicates()

        repeat_returns.to_csv(creds['repeat_returns_path'])
        product_returns.to_csv(creds['product_returns_path'])

        with open(r'config/features.yaml', 'r') as file:
            config = yaml.safe_load(file)
            
        cols_reqd = config['required_columns']
        cat_columns = config['categorical_columns']
        continuous_columns = config['continuous_columns']
            
        train_reqd = train[cols_reqd]

        # One Hot Encoding the required columns
        encoded_data = encode_columns(train_reqd, cat_columns, method='onehot')
        logging.info("One-hot encoding complete.")
        
        # Scaling specified columns
        final_train = scale_data(encoded_data, continuous_columns)
        
        # Prepare feature matrix X and target vector y
        X = final_train.drop('Returned', axis=1)
        y = train['Returned']
        
        X.to_csv(creds['preprocessed_X_path'])
        y.to_csv(creds['preprocessed_y_path'])
        # Ensure y has a single column
        if isinstance(y, pd.DataFrame):
            y = y.squeeze()
        
        MODEL_SAVE_PATH = creds['model_save_path']
        
        # Run the tuning and model saving process
        tune_and_save_model(X, y, MODEL_SAVE_PATH)
        
    except Exception as e:
        logging.error(f"An error occurred in the feature engineering process: {e}")
