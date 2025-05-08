# Diamond Price Prediction

This Jupyter Notebook explores the **Diamond Price Prediction** problem, aiming to build regression models to predict diamond prices based on their attributes.

## Overview

This project involves analyzing a dataset of diamond characteristics and employing various machine learning regression techniques to accurately predict their prices. The notebook covers data loading, exploratory data analysis (EDA), preprocessing, model building, and evaluation.

## Table of Contents

1.  [Data Loading and Initial Exploration](#1-data-loading-and-initial-exploration)
2.  [Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
    * [Carat vs. Price](#carat-vs-price)
    * [Color vs. Price](#color-vs-price)
    * [Clarity vs. Price](#clarity-vs-price)
    * [Cut vs. Price](#cut-vs-price)
    * [Relationship of Numerical Features with Price](#relationship-of-numerical-features-with-price)
    * [Correlation Analysis](#correlation-analysis)
3.  [Data Preprocessing](#3-data-preprocessing)
    * [Handling Categorical Features](#handling-categorical-features)
    * [Splitting Data](#splitting-data)
    * [Feature Scaling](#feature-scaling)
4.  [Model Building and Evaluation](#4-model-building-and-evaluation)
    * [Model Selection](#model-selection)
    * [Model Training and Prediction](#model-training-and-prediction)
    * [Evaluation Metrics](#evaluation-metrics)
    * [Results Display](#results-display)
5.  [Summary of Findings](#summary-of-findings)
6.  [Conclusion](#conclusion)

## 1. Data Loading and Initial Exploration

The notebook starts by importing essential Python libraries for data manipulation and visualization: `pandas`, `numpy`, `seaborn`, and `matplotlib`.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
```

The diamond dataset (`diamonds.csv`) is loaded into a pandas DataFrame:

```python
df = pd.read_csv('diamonds.csv')
```

Initial exploration involves:

* Displaying the first few rows: `df.head()`
* Checking the DataFrame's shape: `df.shape`
* Examining data types and non-null values: `df.info()`
* Calculating descriptive statistics: `df.describe()`

## 2. Exploratory Data Analysis (EDA)

This section visualizes the relationships between different diamond attributes and their prices.

### Carat vs. Price

Distribution of diamond weights:

```python
plt.figure(figsize=(8, 6))
sns.histplot(df['carat'], bins=50, kde=True)
plt.title('Distribution of Diamond Carat')
plt.xlabel('Carat')
plt.ylabel('Frequency')
plt.show()
```

Relationship between carat and price:

```python
plt.figure(figsize=(8, 6))
sns.regplot(x='carat', y='price', data=df)
plt.title('Carat vs. Price')
plt.xlabel('Carat')
plt.ylabel('Price')
plt.show()
```

### Color vs. Price

Distribution of diamond colors:

```python
plt.figure(figsize=(8, 6))
sns.countplot(x='color', data=df, order=df['color'].value_counts().index)
plt.title('Distribution of Diamond Colors')
plt.xlabel('Color')
plt.ylabel('Count')
plt.show()
```

Average price per color:

```python
plt.figure(figsize=(8, 6))
sns.barplot(x='color', y='price', data=df, order=df['color'].value_counts().index)
plt.title('Average Price per Diamond Color')
plt.xlabel('Color')
plt.ylabel('Price')
plt.show()
```

### Clarity vs. Price

Distribution of diamond clarity grades:

```python
plt.figure(figsize=(8, 6))
sns.countplot(x='clarity', data=df, order=df['clarity'].value_counts().index)
plt.title('Distribution of Diamond Clarity Grades')
plt.xlabel('Clarity')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

Average price per clarity:

```python
plt.figure(figsize=(8, 6))
sns.barplot(x='clarity', y='price', data=df, order=df['clarity'].value_counts().index)
plt.title('Average Price per Diamond Clarity')
plt.xlabel('Clarity')
plt.ylabel('Price')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

### Cut vs. Price

Distribution of diamond cut qualities:

```python
plt.figure(figsize=(8, 6))
sns.countplot(x='cut', data=df, order=df['cut'].value_counts().index)
plt.title('Distribution of Diamond Cut Qualities')
plt.xlabel('Cut')
plt.ylabel('Count')
plt.show()
```

Average price per cut:

```python
plt.figure(figsize=(8, 6))
sns.barplot(x='cut', y='price', data=df, order=df['cut'].value_counts().index)
plt.title('Average Price per Diamond Cut')
plt.xlabel('Cut')
plt.ylabel('Price')
plt.show()
```

### Relationship of Numerical Features with Price

Scatter plots showing the relationship between numerical features ('x', 'y', 'z', 'depth', 'table') and 'price':

```python
numerical_features = ['x', 'y', 'z', 'depth', 'table']
for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.regplot(x=feature, y='price', data=df)
    plt.title(f'{feature} vs. Price')
    plt.xlabel(feature)
    plt.ylabel('Price')
    plt.show()
```

### Correlation Analysis

Heatmap of the correlation matrix:

```python
correlation_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Diamond Features')
plt.show()
```

## 3. Data Preprocessing

This section prepares the data for model training.

### Handling Categorical Features

Categorical features ('clarity', 'color', 'cut') are converted to numerical using Label Encoding:

```python
label_encoder = LabelEncoder()
df['clarity'] = label_encoder.fit_transform(df['clarity'])
df['color'] = label_encoder.fit_transform(df['color'])
df['cut'] = label_encoder.fit_transform(df['cut'])
```

### Splitting Data

The dataset is split into training (80%) and testing (20%) sets:

```python
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
```

### Feature Scaling

Numerical features are scaled using StandardScaler:

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## 4. Model Building and Evaluation

This section trains and evaluates various regression models.

### Model Selection

A list of regression models to be evaluated:

```python
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Elastic Net Regression": ElasticNet(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Random Forest Regressor": RandomForestRegressor(random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
    "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
    "SVR": SVR(),
    "K-Nearest Neighbors Regressor": KNeighborsRegressor(),
    "XGBoost Regressor": XGBRegressor(random_state=42),
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor(random_state=42)
}
```

### Model Training and Prediction

Each model is trained on the scaled training data, and predictions are made on the scaled test data.

```python
results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r_squared = r2_score(y_test, y_pred)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    mean_cv_r2 = np.mean(cv_scores)

    results.append([name, rmse, r_squared, mean_cv_r2])
```

### Evaluation Metrics

The performance of each model is evaluated using Root Mean Squared Error (RMSE), R-squared, and the mean R-squared from 5-fold cross-validation.

### Results Display

The evaluation results are displayed in a pandas DataFrame:

```python
results_df = pd.DataFrame(results, columns=['Model', 'RMSE', 'R-squared', 'Mean CV R-squared'])
print(results_df.sort_values(by='R-squared', ascending=False))
```

## 5. Summary of Findings

Based on the displayed output, ensemble methods like Random Forest Regressor, Gradient Boosting Regressor, XGBoost Regressor, and HistGradientBoostingRegressor generally achieve the highest R-squared values and the lowest RMSE values, indicating better predictive performance compared to linear models and simpler tree-based models. The mean cross-validation R-squared scores provide a more reliable estimate of how well these models generalize to unseen data.

## 6. Conclusion

This notebook demonstrates a comprehensive approach to diamond price prediction using various machine learning regression techniques. The EDA provides valuable insights into the relationships between diamond attributes and price. The preprocessing steps ensure the data is suitable for model training. The evaluation results highlight the effectiveness of ensemble learning methods for this regression task. Further steps could involve hyperparameter tuning for the top-performing models to potentially improve their accuracy.
