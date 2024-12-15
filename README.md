# Final Project Programming I Real Estate Toolkit Analysis
## Contributions

This project was a collaborative effort by:

Àlex Muñoz (U199014), David Almirall (U197560), Gorka Bravo (U197568)

## Overview

The Real Estate Analysis Toolkit is a comprehensive Python package designed to analyze housing market data and simulate real estate market dynamics using agent-based modeling. This project demonstrates the application of modern data analysis libraries, machine learning pipelines, and clean coding principles to solve real-world problems.

The project was completed using the Ames Housing dataset, which provides detailed features for analyzing house prices, such as structural attributes, neighborhood characteristics, and utility details.

## Key Features

### Data Loading and Cleaning:

  -Robust data loader to handle CSV files.

  -Data cleaning and preprocessing methods including handling missing values and renaming columns for clarity.

### Descriptive Analytics:

  -Compute metrics such as averages, medians, percentiles, and mode for numerical and categorical variables.

  -Perform correlation analysis and exploratory data analysis with interactive visualizations.

### Agent-Based Modeling:

  -Simulate market dynamics with agents representing buyers and houses.

  -Classes for houses, consumers, and housing markets.

  -Market clearing mechanisms with multiple configurations.

### Machine Learning:

-Predictive modeling with Scikit-Learn pipelines.

-Baseline models such as Linear Regression and Random Forest.

-Feature engineering and model evaluation with various metrics.

## Project Structure

```
real_estate_toolkit/
├── pyproject.toml
├── README.md
├── .venv/
└── src/
    └── real_estate_toolkit/
        ├── __init__.py
        ├── data/
        │   ├── __init__.py
        │   ├── loader.py
        │   ├── cleaner.py
        │   └── descriptor.py
        ├── agent_based_model/
        │   ├── __init__.py
        │   ├── consumers.py
        │   ├── houses.py
        │   ├── house_market.py
        │   └── simulation.py
        ├── analytics/
        │   ├── __init__.py
        │   ├── outputs/
        │   └── exploratory.py
        ├── ml_models/
        │   ├── __init__.py
        │   └── predictor.py
        └── main.py
```


## Installation and Usage

###Prerequisites

-Ensure you have Python 3.8 or higher installed. Use poetry for dependency management.

Installation

Clone this repository:
```
git clone https://github.com/username/real_estate_toolkit.git
cd real_estate_toolkit
```
Install dependencies using Poetry:
```
poetry install
poetry shell
```

## Highlights

### Data Analysis

Exploratory Data Analysis:

-Interactive visualizations using Plotly.

-Price distributions and neighborhood comparisons.

Feature Engineering:

-Handling missing values.

-Standardizing and one-hot encoding categorical variables.

### Agent-Based Modeling

-Simulate market dynamics with customizable agents.

-Analyze ownership and availability rates under various market conditions.

### Machine Learning

-Baseline models: Linear Regression and Random Forest Regressor.

-Evaluation metrics: RMSE, MAE, R^2, and MAPE.

-Output predictions in a Kaggle-compatible format.


## Acknowledgments
Dataset: Ames Housing Dataset from the Kaggle competition "House Prices - Advanced Regression Techniques."

Guidance: José Fernando Moreno.
