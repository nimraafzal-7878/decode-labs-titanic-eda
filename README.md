
# Titanic Dataset - Advanced EDA & Feature Engineering

## Project Overview
This project performs Exploratory Data Analysis (EDA),
data cleaning, outlier detection, feature engineering,
and categorical encoding on the Titanic dataset.

## Dataset
- Dataset: Titanic
- Total Rows: 891
- Original Columns: 12

## Tools & Libraries
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Tasks Completed

### 1. Exploratory Data Analysis
- Examined dataset structure
- Checked missing values
- Generated statistical summaries
- Visualized Age distribution
- Created Fare boxplot

### 2. Outlier Detection
- Applied the IQR method
- Identified 116 Fare outliers
- Created Fare_Capped using IQR boundaries

### 3. Missing Value Handling
- Age: Median Imputation
- Embarked: Mode Imputation
- Cabin: Filled with "Unknown"
- Compared Median and KNN Imputation

### 4. Feature Engineering
Created three new features:
- FamilySize
- IsAlone
- Title

### 5. Categorical Encoding
Converted categorical features into numeric
columns using one-hot encoding.

### 6. Correlation Analysis
Generated a correlation heatmap to analyze
relationships between features and survival.

### 7. Final Validation
- Missing Values: 0
- Original Dataset Duplicates: 0
- Duplicate Passenger IDs: 0
- Encoded Dataset Shape: 891 rows x 30 columns

Note: 113 identical feature rows were found after
encoding. These were retained because the original
passenger records have unique IDs.

## Output Files
- titanic_cleaned.csv
- titanic_ml_ready.csv

## How to Run
Run the following command:

python main.py
