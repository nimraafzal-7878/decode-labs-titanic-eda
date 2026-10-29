import pandas as pd

# Load Titanic dataset from local CSV file
df = pd.read_csv("titanic_dataset.csv")

# Display first 5 rows
print(df.head())

# Display dataset size
print("Dataset Shape:", df.shape)

# Display all column names
print("\nDataset Columns:")
print(df.columns.tolist())


# Step 6: Exploratory Data Analysis (EDA)

# Display dataset information
print("\n----- DATASET INFORMATION -----")
df.info()

# Display missing values in each column
print("\n----- MISSING VALUES -----")
print(df.isnull().sum())

print("\n----- STATISTICAL SUMMARY -----")
print(df.describe())

# Step 7: Exploratory Data Visualization

import matplotlib.pyplot as plt
import seaborn as sns

# Age Distribution Graph
plt.figure(figsize=(10, 5))

sns.histplot(df["Age"], bins=20, kde=True)

plt.title("Age Distribution of Titanic Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.show()


# Step 7.2: Fare Boxplot

plt.figure(figsize=(10, 5))

sns.boxplot(x=df["Fare"])

plt.title("Fare Distribution of Titanic Passengers")
plt.xlabel("Ticket Fare")

plt.show()


# Step 8: Outlier Detection Using IQR

# Calculate Q1 and Q3
Q1 = df["Fare"].quantile(0.25)
Q3 = df["Fare"].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1

# Calculate outlier boundaries
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = df[
    (df["Fare"] < lower_bound) |
    (df["Fare"] > upper_bound)
]

# Display results
print("\n----- IQR OUTLIER DETECTION -----")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)
print("Total Outliers:", len(outliers))

# Display first 10 outliers
print("\nFirst 10 Outliers:")
print(outliers[["PassengerId", "Fare"]].head(10))


# Step 9: Outlier Treatment

# Create a copy of original dataset
df_clean = df.copy()

# Cap Fare outliers
df_clean["Fare_Capped"] = df_clean["Fare"].clip(
    lower=max(0, lower_bound),
    upper=upper_bound
)


# Step 10: Missing Values Handling

print("\n----- BEFORE CLEANING -----")
print(df_clean.isnull().sum())

# Fill missing Age values
df_clean["Age"] = df_clean["Age"].fillna(
    df_clean["Age"].median()
)

# Fill missing Embarked values
df_clean["Embarked"] = df_clean["Embarked"].fillna(
    df_clean["Embarked"].mode()[0]
)

# Fill missing Cabin values
df_clean["Cabin"] = df_clean["Cabin"].fillna("Unknown")

print("\n----- AFTER CLEANING -----")
print(df_clean.isnull().sum())


# Step 11: KNN Imputation

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# Select numerical columns for KNN
features = df[["Age", "Pclass", "SibSp", "Parch", "Fare"]].copy()

# Scale numerical values
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# Apply KNN Imputation
imputer = KNNImputer(n_neighbors=5)
knn_data = imputer.fit_transform(scaled_data)

# Convert values back to original scale
knn_data = scaler.inverse_transform(knn_data)

# Store KNN estimated ages
df_clean["Age_KNN"] = knn_data[:, 0]

# Compare Median and KNN
print("\n----- MEDIAN VS KNN IMPUTATION -----")
print("Original Missing Ages:", df["Age"].isnull().sum())
print("Missing Ages After KNN:", df_clean["Age_KNN"].isnull().sum())

print("\nFirst 10 Originally Missing Ages:")
print(
    df_clean.loc[
        df["Age"].isnull(),
        ["Age", "Age_KNN"]
    ].head(10)
)


# Step 11.2: Mean Imputation (Comparison with Median and KNN)

age_mean = df["Age"].fillna(df["Age"].mean())

print("\n----- MEAN IMPUTATION -----")
print("Mean Age:", df["Age"].mean())
print("Missing Ages After Mean:", age_mean.isnull().sum())

comparison = pd.DataFrame({
    "Mean": age_mean,
    "Median": df_clean["Age"],
    "KNN": df_clean["Age_KNN"]
})

print("\n----- MEAN VS MEDIAN VS KNN -----")
print(comparison.loc[df["Age"].isnull()].head(10))


# Step 12: Feature Engineering

# Feature 1: Calculate Family Size
df_clean["FamilySize"] = (
    df_clean["SibSp"] + df_clean["Parch"] + 1
)

# Feature 2: Check if passenger is alone
df_clean["IsAlone"] = (
    df_clean["FamilySize"] == 1
).astype(int)

# Feature 3: Extract Title from Name
df_clean["Title"] = df_clean["Name"].str.extract(
    r",\s*([^.]*)\.", expand=False
).str.strip()

# Display newly created features
print("\n----- NEW FEATURES -----")

print(
    df_clean[
        ["Name", "FamilySize", "IsAlone", "Title"]
    ].head(10)
)

# Check total columns
print("\nTotal Columns:", df_clean.shape[1])


# Step 13: Categorical Encoding

# Create a separate dataset for ML preparation
df_encoded = df_clean.drop(
    columns=["PassengerId", "Name", "Ticket", "Cabin", "Fare", "Age_KNN"]
).copy()

# Convert categorical columns into numerical columns
df_encoded = pd.get_dummies(
    df_encoded,
    columns=["Sex", "Embarked", "Title"],
    dtype=int
)

# Display encoded dataset
print("\n----- ENCODED DATASET -----")
print(df_encoded.head())

# Display dataset shape
print("\nEncoded Dataset Shape:", df_encoded.shape)

# Check if any text columns remain
print("\nRemaining Text Columns:")
print(df_encoded.select_dtypes(include="object").columns.tolist())


# Step 14: Correlation Analysis

# Calculate correlation between numerical features
correlation_matrix = df_encoded.corr()

# Check correlation with Survived
survival_corr = correlation_matrix["Survived"].drop("Survived")

# Sort by strongest relationship
survival_corr = survival_corr.reindex(
    survival_corr.abs().sort_values(ascending=False).index
)

print("\n----- TOP 10 CORRELATIONS WITH SURVIVAL -----")
print(survival_corr.head(10))


# Create Correlation Heatmap

# Select top 10 features plus Survived
top_features = ["Survived"] + survival_corr.head(10).index.tolist()

plt.figure(figsize=(12, 8))

sns.heatmap(
    df_encoded[top_features].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Titanic Feature Correlation Heatmap")
plt.tight_layout()
plt.show()


# Step 15: Final Dataset Validation

print("\n----- FINAL DATASET VALIDATION -----")

# Check missing values
print("Missing Values:", df_encoded.isnull().sum().sum())

# Check duplicate rows
print("Duplicate Rows:", df_encoded.duplicated().sum())

# Check dataset size
print("Final Dataset Shape:", df_encoded.shape)

# Check remaining text columns
print("Text Columns:",
      df_encoded.select_dtypes(include="object").columns.tolist())


# Step 16: Save Final Datasets

# Save cleaned dataset
df_clean.to_csv("titanic_cleaned.csv", index=False)

# Save encoded ML-ready dataset
df_encoded.to_csv("titanic_ml_ready.csv", index=False)

print("\n----- FILES SAVED SUCCESSFULLY -----")
print("titanic_cleaned.csv")
print("titanic_ml_ready.csv")


# Step 17: Verify Duplicate Rows

print("\n----- DUPLICATE VERIFICATION -----")

print("Original Dataset Duplicates:", df.duplicated().sum())

print("Duplicate Passenger IDs:",
      df["PassengerId"].duplicated().sum())

print("Encoded Dataset Duplicates:",
      df_encoded.duplicated().sum())
