from typing import List, Dict, Any, Optional, Tuple
import os
from pathlib import Path

import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)


class HousePricePredictor:
    def __init__(self, train_data_path: str, test_data_path: str):
        """
        Initialize the predictor class with paths to the training and testing datasets.
        
        Args:
            train_data_path (str): Path to the training dataset CSV file.
            test_data_path (str): Path to the testing dataset CSV file.
        
        Attributes:
            - self.train_data: Polars DataFrame for the training dataset.
            - self.test_data: Polars DataFrame for the testing dataset.
        """
        self.train_data = pl.read_csv(train_data_path)
        self.test_data = pl.read_csv(test_data_path)
        
        # Attributes to store after preparation and training
        self.preprocessor = None
        self.trained_models = {}
        self.numeric_features = []
        self.categorical_features = []
        self.target_column = "SalePrice"

    def clean_data(self):
        """
        Perform comprehensive data cleaning on the training and testing datasets.
        
        Tasks:
        1. Handle Missing Values:
            - For numeric columns: fill missing values with mean.
            - For categorical columns: fill missing values with "Missing".
        2. Ensure Correct Data Types:
            - Convert numeric-looking columns to float.
            - Convert other columns to categorical (string).
        3. (Optional) Drop Unnecessary Columns:
            - Identify and remove columns with too many missing values or irrelevant information.
        
        Notes:
            The exact strategy can vary. Here we take a generic approach:
            - Convert columns that are entirely numeric to floats.
            - Convert other columns to string.
            - Impute missing values.
        """
        
        def determine_col_types(df: pl.DataFrame) -> Tuple[List[str], List[str]]:
            # We'll guess numeric vs categorical by trying to parse as float
            numeric_cols = []
            categorical_cols = []
            for col in df.columns:
                # Skip target column type changes if it's numeric
                if col == self.target_column:
                    # If SalePrice is present in this df and is numeric
                    # assume numeric
                    if df[col].dtype in [pl.Float64, pl.Int64]:
                        numeric_cols.append(col)
                    else:
                        # Try to cast to float if possible
                        try:
                            _ = df[col].cast(pl.Float64)
                            numeric_cols.append(col)
                        except:
                            # If not castable, treat as categorical
                            categorical_cols.append(col)
                    continue

                # Try casting to float
                try:
                    df[col].cast(pl.Float64)  # Just a test cast, not assigning
                    numeric_cols.append(col)
                except:
                    categorical_cols.append(col)

            return numeric_cols, categorical_cols

        def impute_and_cast(df: pl.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> pl.DataFrame:
            # For numeric columns: fill missing with mean
            # For categorical columns: fill missing with "Missing"
            # Then cast columns to appropriate type

            # Handle numeric columns
            for col in numeric_cols:
                # Compute mean, fill nulls
                mean_val = df[col].mean()
                df = df.with_columns(
                    pl.when(pl.col(col).is_null()).then(mean_val).otherwise(pl.col(col)).alias(col)
                )
                # Cast to float64
                df = df.with_columns(pl.col(col).cast(pl.Float64))

            # Handle categorical columns
            for col in categorical_cols:
                df = df.with_columns(
                    pl.when(pl.col(col).is_null()).then("Missing").otherwise(pl.col(col)).alias(col)
                )
                df = df.with_columns(pl.col(col).cast(pl.Utf8))

            return df

        # First, identify columns as numeric or categorical in training data
        train_numeric_cols, train_categorical_cols = determine_col_types(self.train_data)
        test_numeric_cols, test_categorical_cols = determine_col_types(self.test_data)

        # Ensure consistency of columns between train and test:
        # If a column appears numeric in train but not in test or vice versa,
        # we unify by intersection. Columns must have consistent interpretation.
        numeric_set = set(train_numeric_cols).intersection(set(test_numeric_cols))
        categorical_set = set(train_categorical_cols).intersection(set(test_categorical_cols))

        # Columns exclusive to train or test can cause issues. Let's handle by:
        # If a column is numeric in train but doesn't exist in test (or vice versa),
        # drop it from consideration. Similarly for categorical.
        # (This is a simplistic approach.)
        # Also, if a column is numeric in train and categorical in test, we default
        # to treating it as categorical in both.
        for col in set(train_numeric_cols + test_numeric_cols):
            if col in train_numeric_cols and col in test_categorical_cols:
                # Conflict: treat as categorical
                numeric_set.discard(col)
                categorical_set.add(col)
            elif col in test_numeric_cols and col in train_categorical_cols:
                # Conflict: treat as categorical
                numeric_set.discard(col)
                categorical_set.add(col)

        # Now we have consistent sets of numeric and categorical columns
        self.numeric_features = list(numeric_set)
        # Exclude target from categorical if ended there
        if self.target_column in self.numeric_features:
            self.numeric_features.remove(self.target_column)
        if self.target_column in categorical_set:
            categorical_set.discard(self.target_column)
        self.categorical_features = list(categorical_set)

        # Impute and cast both datasets
        self.train_data = impute_and_cast(self.train_data, self.numeric_features + [self.target_column], self.categorical_features)
        self.test_data = impute_and_cast(self.test_data, self.numeric_features, self.categorical_features)

    def prepare_features(self, target_column: str = 'SalePrice', selected_predictors: List[str] = None):
        """
        Prepare the dataset for machine learning by separating features and the target variable, 
        and preprocessing them for training and testing.

        Args:
            target_column (str): Name of the target variable column. Default is 'SalePrice'.
            selected_predictors (List[str]): Specific columns to use as predictors. 
                                            If None, use all columns except the target.

        Returns:
            - X_train, X_test, y_train, y_test: Training and testing sets.
        """

        self.target_column = target_column

        if selected_predictors is None:
            # Use all columns except target
            predictors = [col for col in self.train_data.columns if col != self.target_column]
        else:
            predictors = selected_predictors

        # Separate features and target
        X = self.train_data.select(predictors)
        y = self.train_data.select(self.target_column).to_series()

        # Identify numeric and categorical from the cleaned data
        numeric_features = [col for col in predictors if col in self.numeric_features]
        categorical_features = [col for col in predictors if col in self.categorical_features]

        # Define the transformers
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        self.preprocessor = preprocessor

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X.to_pandas(), y.to_pandas(), test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

    def train_baseline_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Train and evaluate baseline machine learning models for house price prediction.
        
        Models:
        - Linear Regression
        - Advanced Model: RandomForestRegressor or GradientBoostingRegressor

        Returns:
            A dictionary structured like:
                {
                    "Linear Regression": 
                        { 
                            "metrics": {"MSE": ..., "R2": ..., "MAE": ..., "MAPE": ...},
                            "model": (model object)
                        },
                    "Advanced Model":
                        { 
                            "metrics": {"MSE": ..., "R2": ..., "MAE": ..., "MAPE": ...},
                            "model": (model object)
                        }
                }
        """
        # We must have already run prepare_features to get X_train, X_test, y_train, y_test
        # For flexibility, let's just call prepare_features here without arguments
        X_train, X_test, y_train, y_test = self.prepare_features(target_column=self.target_column)

        # Models
        linear_model = LinearRegression()
        advanced_model = GradientBoostingRegressor(random_state=42)

        # Pipelines
        lin_pipeline = Pipeline(steps=[('preprocessor', self.preprocessor),
                                       ('model', linear_model)])

        adv_pipeline = Pipeline(steps=[('preprocessor', self.preprocessor),
                                       ('model', advanced_model)])

        # Train models
        lin_pipeline.fit(X_train, y_train)
        adv_pipeline.fit(X_train, y_train)

        # Evaluate models
        def evaluate_model(model, X_tr, y_tr, X_te, y_te):
            y_pred_train = model.predict(X_tr)
            y_pred_test = model.predict(X_te)
            metrics = {
                "MSE_train": mean_squared_error(y_tr, y_pred_train),
                "MSE_test": mean_squared_error(y_te, y_pred_test),
                "MAE_train": mean_absolute_error(y_tr, y_pred_train),
                "MAE_test": mean_absolute_error(y_te, y_pred_test),
                "R2_train": r2_score(y_tr, y_pred_train),
                "R2_test": r2_score(y_te, y_pred_test),
                "MAPE_train": mean_absolute_percentage_error(y_tr, y_pred_train),
                "MAPE_test": mean_absolute_percentage_error(y_te, y_pred_test)
            }
            return metrics

        lin_metrics = evaluate_model(lin_pipeline, X_train, y_train, X_test, y_test)
        adv_metrics = evaluate_model(adv_pipeline, X_train, y_train, X_test, y_test)

        self.trained_models = {
            "Linear Regression": {
                "metrics": lin_metrics,
                "model": lin_pipeline
            },
            "Advanced Model": {
                "metrics": adv_metrics,
                "model": adv_pipeline
            }
        }

        return self.trained_models

    def forecast_sales_price(self, model_type: str = 'LinearRegression'):
        """
        Use the trained model to forecast house prices on the test dataset.
        
        Args:
            model_type (str): Type of model to use for forecasting. Default is 'LinearRegression'. Other option: 'Advanced'.
        
        Tasks:
            1. Select the Desired Model:
                - Ensure the model type is trained and available.
            2. Generate Predictions:
                - Use the selected model to predict house prices for the test set.
            3. Create a Submission File:
                - Save predictions in CSV with columns: "Id" (from test data) and "SalePrice".
            4. Save the File at:
                src/real_estate_toolkit/ml_models/outputs/submission.csv
        """

        if not self.trained_models:
            raise ValueError("No trained models found. Please run `train_baseline_models` first.")

        # Choose model
        if model_type == 'LinearRegression':
            model_key = "Linear Regression"
        else:
            model_key = "Advanced Model"

        if model_key not in self.trained_models:
            raise ValueError(f"Model type {model_type} not trained or not recognized.")

        model_pipeline = self.trained_models[model_key]['model']

        # Prepare test_data for predictions
        # We must apply the same preprocessing. The pipeline will handle it,
        # but we must ensure the test_data has the same columns as training features.
        
        # Identify the predictors used in training
        # The pipeline's first step is the preprocessor that was fit on training data
        # The original selected predictors are those passed in prepare_features.
        # After training_baseline_models, we have X_train from prepare_features.
        # Let's just get them from the pipeline steps.
        # The pipeline was fit on X_train which had certain columns.
        # We can retrieve these columns:
        trained_feature_names = model_pipeline['preprocessor'].transformers_[0][2] + model_pipeline['preprocessor'].transformers_[1][2]
        
        # trained_feature_names should match the predictors chosen. We used all except target.
        # Ensure test_data has all these columns. If not, fill them with missing.
        for col in trained_feature_names:
            if col not in self.test_data.columns:
                # Add missing column
                self.test_data = self.test_data.with_columns(pl.lit("Missing").alias(col) if col in self.categorical_features else pl.lit(None).alias(col))

        # Select the features from test_data
        X_final_test = self.test_data.select(trained_feature_names).to_pandas()

        # Predict
        predictions = model_pipeline.predict(X_final_test)

        # Assume test_data has an 'Id' column to identify rows. If not, we create one.
        # In many Kaggle competitions like House Prices, test dataset has an 'Id' column.
        # If not present, we create a dummy one.
        if 'Id' not in self.test_data.columns:
            # Create a dummy Id column
            self.test_data = self.test_data.with_columns(
                (pl.Series(name="Id", values=range(1, self.test_data.height + 1)))
            )

        submission_df = pl.DataFrame({
            "Id": self.test_data["Id"],
            "SalePrice": pl.Series(values=predictions)
        })

        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        submission_path = output_dir / "submission.csv"
        submission_df.write_csv(str(submission_path))

