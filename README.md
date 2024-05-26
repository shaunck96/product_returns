# Customer Return Prediction Pipeline

This document describes the implementation of a data pipeline to predict customer returns using machine learning. The pipeline involves data loading, preprocessing, feature engineering, model training, evaluation, and generating summary reports.

## Table of Contents
- [Overview](#overview)
- [Data Loading](#data-loading)
- [Data Quality Checks](#data-quality-checks)
- [Feature Engineering](#feature-engineering)
- [Data Encoding and Scaling](#data-encoding-and-scaling)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Prediction and Evaluation](#prediction-and-evaluation)
- [Summary Report Generation](#summary-report-generation)

## Overview
This project involves building a machine learning model to predict whether a customer will return a product. The workflow includes loading datasets, performing data quality checks, engineering features, encoding and scaling data, training and evaluating models, and generating predictions. Finally, a summary report is generated with visualizations and statistics.

## Data Loading

### `DataLoader` Class
This class handles loading the training and testing datasets from CSV files.

#### Attributes
- `train_path` (str): Path to the training dataset CSV file.
- `test_path` (str): Path to the testing dataset CSV file.
- `train` (DataFrame): DataFrame containing the training data.
- `test` (DataFrame): DataFrame containing the testing data.

#### Methods
- `__init__(self, train_path='train.csv', test_path='test.csv')`: Initializes the DataLoader class, loading the train and test datasets.
- `check_data_quality(self)`: Executes all data quality checks on the datasets.
- `check_dates(self)`: Checks for date-related errors.
- `check_strings(self)`: Checks for issues in string columns.
- `check_floats(self)`: Checks for issues in float columns.
- `convert_column_types(self)`: Converts columns to their appropriate data types.
- `load_data(self)`: Returns the training and testing datasets.

## Data Quality Checks

### `check_dates` Method
This method checks for date-related errors such as incorrect formats and future dates.

### `check_strings` Method
This method checks for issues in string columns like missing values and duplicate IDs.

### `check_floats` Method
This method checks for issues in float columns like negative values and unreasonable percentages.

## Feature Engineering

### `load_and_prepare_data` Function
This function loads and prepares data for feature engineering.

### `add_calculated_columns` Function
This function adds calculated columns to the dataset, such as:
- `msrp`: Calculated MSRP based on Purchase Price and Discount Percentage.
- `RepeatReturnFlag`: Indicates if a customer has multiple returns.
- `MultiItemOrder`: Indicates if an order contains multiple items.
- `Season`: Adds a season column based on the order date.
- `CustomerAge`: Calculates the customer's age at the time of the order.
- `Holiday`: Indicates if the order date is a US federal holiday.
- `DaysSinceFirstOrder`: Days since the customer's first order.
- `CustomerLifetimeValue`: Total purchase value of a customer.
- `OrderFrequency`: Frequency of orders by a customer.
- `ProductReturnRate`: Return rate of products in each department.
- `DayOfWeek`: Day of the week of the order date.
- `RecentReturnRate`: Rolling average of recent returns.
- `PriceSensitivity`: Sensitivity to price discounts.
- `DaysBetweenOrders`: Days between consecutive orders.
- `AvgDaysBetweenOrders`: Average days between orders.

## Data Encoding and Scaling

### `encode_columns` Function
Encodes categorical columns using various methods such as one-hot encoding, label encoding, binary encoding, etc.

### `scale_data` Function
Scales continuous columns using `StandardScaler`.

## Model Training and Evaluation

### `stratified_training` Function
Trains and evaluates multiple models using stratified k-fold cross-validation. The models include:
- Random Forest
- Gradient Boosting
- Extra Trees
- XGBoost
- LightGBM

### `tune_and_save_model` Function
Tunes the best model using `RandomizedSearchCV` and saves the trained model.

### `save_metrics` Function
Saves model performance metrics to a CSV file.

## Prediction and Evaluation

### `load_model` Function
Loads a trained model from a specified path.

### Prediction and Analysis
- Merges historical datasets for repeat returns and product return rates.
- Adds calculated columns to the test dataset.
- Encodes and scales the test data.
- Generates predictions using the loaded model.
- Saves predictions to a CSV file.

### Feature Importance Visualization
- Plots feature importances of the trained model using `matplotlib` and `seaborn`.

## Summary Report Generation

### Summary Statistics
- Generates summary statistics for returned and non-returned customers.
- Saves summary statistics to CSV files.

### Visualizations
- Plots histograms and bar charts for continuous and categorical features, respectively.
- Saves the plots as PNG files.

### PDF Report
- Generates a PDF report with summary statistics and visualizations using `FPDF`.
