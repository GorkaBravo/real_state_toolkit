from typing import List, Dict
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os


class MarketAnalyzer:
    def __init__(self, data_path: str):
        """
        Initialize the analyzer with data from a CSV file.

        Args:
            data_path (str): Path to the Ames Housing dataset CSV file.
        """
    
        self.data_path = data_path
        try:
            self.real_state_data = pl.read_csv(self.data_path)
            print(f"Data loaded successfully from {self.data_path}")
        except Exception as e:
            print(f"Failed to load data from {self.data_path}: {e}")
            raise e
        self.real_state_clean_data = None
        self.output_dir = Path("src/real_estate_toolkit/analytics/outputs/")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def clean_data(self) -> None:
        """
        Perform comprehensive data cleaning:
        
        Tasks implemented:
        1. Identify and handle missing values
            - Drop columns with excessive missing values.
            - Fill numerical columns with median.
            - Fill categorical columns with mode.
        2. Convert columns to appropriate data types if needed.
            - Ensure numeric columns are numeric.
            - Ensure categorical columns are categorized.
        
        Returns:
            Cleaned and preprocessed dataset assigned to self.real_state_clean_data
        """
        df = self.real_state_data.clone()

        # Identify missing values
        missing_counts = df.null_count()
        print("Missing values per column:")
        print(missing_counts)

        # Strategy:
        # - Drop columns with more than 20% missing values
        # - For numerical columns, fill missing with median
        # - For categorical columns, fill missing with mode

        threshold = 0.2 * df.height
        columns_to_drop = missing_counts.filter(pl.col("count") > threshold).select(pl.col("column_name")).to_series().to_list()

        if columns_to_drop:
            print(f"Dropping columns with more than 20% missing values: {columns_to_drop}")
            df = df.drop(columns_to_drop)

        # Identify numerical and categorical columns
        numerical_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype in [pl.Int64, pl.Float64]]
        categorical_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype in [pl.Utf8, pl.Categorical]]

        # Fill missing numerical columns with median
        for col in numerical_cols:
            if df[col].is_null().sum() > 0:
                median_val = df[col].median()
                print(f"Filling missing values in numerical column '{col}' with median: {median_val}")
                df = df.with_column(pl.col(col).fill_null(median_val))

        # Fill missing categorical columns with mode
        for col in categorical_cols:
            if df[col].is_null().sum() > 0:
                mode_val = df[col].mode().first()
                print(f"Filling missing values in categorical column '{col}' with mode: {mode_val}")
                df = df.with_column(pl.col(col).fill_null(mode_val))

        # Convert numerical columns to appropriate types
        for col in numerical_cols:
            df = df.with_column(pl.col(col).cast(pl.Float64))

        # Convert categorical columns to categorical type
        for col in categorical_cols:
            df = df.with_column(pl.col(col).cast(pl.Categorical))

        self.real_state_clean_data = df
        print("Data cleaning completed and stored in 'real_state_clean_data'.")

    def generate_price_distribution_analysis(self) -> pl.DataFrame:
        """
        Analyze sale price distribution using clean data.
        
        Tasks implemented:
        1. Compute basic price statistics and generate another data frame called price_statistics:
            - Mean
            - Median
            - Standard deviation
            - Minimum and maximum prices
        2. Create an interactive histogram of sale prices using Plotly.
        
        Returns:
            Statistical insights dataframe
            Saves Plotly figure for price distribution in src/real_estate_toolkit/analytics/outputs/ folder.
        """
        if self.real_state_clean_data is None:
            raise ValueError("Data not cleaned. Please run 'clean_data()' first.")

        df = self.real_state_clean_data

        # Compute statistics
        price = df["SalePrice"]
        price_statistics = pl.DataFrame({
            "Mean": [price.mean()],
            "Median": [price.median()],
            "Std Dev": [price.std()],
            "Minimum": [price.min()],
            "Maximum": [price.max()]
        })
        print("Price statistics computed:")
        print(price_statistics)

        # Create histogram
        price_pd = price.to_pandas()
        fig = px.histogram(price_pd, x="SalePrice", nbins=50, title="Sale Price Distribution",
                           labels={"SalePrice": "Sale Price"}, 
                           hover_data=True)
        fig.update_layout(bargap=0.1)

        # Save figure
        histogram_path = self.output_dir / "price_distribution_histogram.html"
        fig.write_html(str(histogram_path))
        print(f"Price distribution histogram saved to {histogram_path}")

        return price_statistics

    def neighborhood_price_comparison(self) -> pl.DataFrame:
        """
        Create a boxplot comparing house prices across different neighborhoods.
        
        Tasks implemented:
        1. Group data by neighborhood
        2. Calculate price statistics for each neighborhood
        3. Create Plotly boxplot with:
            - Median prices
            - Price spread
            - Outliers
        
        Returns:
            Neighborhood statistics dataframe
            Saves Plotly figure for neighborhood price comparison in src/real_estate_toolkit/analytics/outputs/ folder.
        """
        if self.real_state_clean_data is None:
            raise ValueError("Data not cleaned. Please run 'clean_data()' first.")

        df = self.real_state_clean_data

        # Ensure 'Neighborhood' and 'SalePrice' columns exist
        if "Neighborhood" not in df.columns or "SalePrice" not in df.columns:
            raise ValueError("Required columns 'Neighborhood' and 'SalePrice' not found in data.")

        # Group by Neighborhood and compute statistics
        neighborhood_stats = df.groupby("Neighborhood").agg([
            pl.col("SalePrice").mean().alias("MeanPrice"),
            pl.col("SalePrice").median().alias("MedianPrice"),
            pl.col("SalePrice").std().alias("StdDevPrice"),
            pl.col("SalePrice").min().alias("MinPrice"),
            pl.col("SalePrice").max().alias("MaxPrice")
        ]).sort("MedianPrice", reverse=True)

        print("Neighborhood price statistics computed:")
        print(neighborhood_stats)

        # Create boxplot
        neighborhood_pd = df.select(["Neighborhood", "SalePrice"]).to_pandas()
        fig = px.box(neighborhood_pd, x="Neighborhood", y="SalePrice",
                     title="House Price Comparison Across Neighborhoods",
                     labels={"SalePrice": "Sale Price", "Neighborhood": "Neighborhood"},
                     points="outliers")
        fig.update_layout(xaxis={'categoryorder':'total ascending'})

        # Save figure
        boxplot_path = self.output_dir / "neighborhood_price_comparison_boxplot.html"
        fig.write_html(str(boxplot_path))
        print(f"Neighborhood price comparison boxplot saved to {boxplot_path}")

        return neighborhood_stats

    def feature_correlation_heatmap(self, variables: List[str]) -> None:
        """
        Generate a correlation heatmap for variables input.
        
        Tasks implemented:
        1. Pass a list of numerical variables
        2. Compute correlation matrix and plot it
        
        Args:
            variables (List[str]): List of variables to correlate
        
        Returns:
            Saves Plotly figure for correlation heatmap in src/real_estate_toolkit/analytics/outputs/ folder.
        """
        if self.real_state_clean_data is None:
            raise ValueError("Data not cleaned. Please run 'clean_data()' first.")

        df = self.real_state_clean_data.select(variables)

        # Check if all variables are numeric
        for var in variables:
            if df[var].dtype not in [pl.Float64, pl.Int64]:
                raise ValueError(f"Variable '{var}' is not numeric and cannot be correlated.")

        # Compute correlation matrix
        corr_matrix = df.to_pandas().corr()
        print("Correlation matrix computed:")
        print(corr_matrix)

        # Create heatmap
        fig = px.imshow(corr_matrix, text_auto=True, 
                        title="Feature Correlation Heatmap",
                        labels=dict(color="Correlation"),
                        x=variables,
                        y=variables)
        fig.update_layout(width=800, height=700)

        # Save figure
        heatmap_path = self.output_dir / "feature_correlation_heatmap.html"
        fig.write_html(str(heatmap_path))
        print(f"Feature correlation heatmap saved to {heatmap_path}")

    def create_scatter_plots(self) -> Dict[str, go.Figure]:
        """
        Create scatter plots exploring relationships between key features.
        
        Scatter plots created:
        1. House price vs. Total square footage
        2. Sale price vs. Year built
        3. Overall quality vs. Sale price
        
        Tasks implemented:
        - Use Plotly Express for creating scatter plots
        - Add trend lines
        - Include hover information
        - Color-code points based on a categorical variable (e.g., Neighborhood)
        - Save them in src/real_estate_toolkit/analytics/outputs/ folder.
        
        Returns:
            Dictionary of Plotly Figure objects for different scatter plots. 
        """
        if self.real_state_clean_data is None:
            raise ValueError("Data not cleaned. Please run 'clean_data()' first.")

        df = self.real_state_clean_data

        scatter_plots = {}

        # 1. House price vs. Total square footage (Assuming 'TotalSF' exists)
        if "TotalSF" in df.columns and "SalePrice" in df.columns:
            total_sf_pd = df.select(["TotalSF", "SalePrice", "Neighborhood"]).to_pandas()
            fig1 = px.scatter(total_sf_pd, x="TotalSF", y="SalePrice",
                             color="Neighborhood",
                             trendline="ols",
                             title="Sale Price vs. Total Square Footage",
                             labels={"TotalSF": "Total Square Footage", "SalePrice": "Sale Price"},
                             hover_data=["Neighborhood"])
            scatter_plots["SalePrice_vs_TotalSF"] = fig1
            scatter_sf_path = self.output_dir / "sale_price_vs_total_sf_scatter.html"
            fig1.write_html(str(scatter_sf_path))
            print(f"Scatter plot 'Sale Price vs. Total Square Footage' saved to {scatter_sf_path}")
        else:
            print("Columns 'TotalSF' or 'SalePrice' not found. Skipping 'Sale Price vs. Total Square Footage' scatter plot.")

        # 2. Sale price vs. Year built
        if "YearBuilt" in df.columns and "SalePrice" in df.columns:
            year_built_pd = df.select(["YearBuilt", "SalePrice", "Neighborhood"]).to_pandas()
            fig2 = px.scatter(year_built_pd, x="YearBuilt", y="SalePrice",
                             color="Neighborhood",
                             trendline="ols",
                             title="Sale Price vs. Year Built",
                             labels={"YearBuilt": "Year Built", "SalePrice": "Sale Price"},
                             hover_data=["Neighborhood"])
            scatter_plots["SalePrice_vs_YearBuilt"] = fig2
            scatter_year_path = self.output_dir / "sale_price_vs_year_built_scatter.html"
            fig2.write_html(str(scatter_year_path))
            print(f"Scatter plot 'Sale Price vs. Year Built' saved to {scatter_year_path}")
        else:
            print("Columns 'YearBuilt' or 'SalePrice' not found. Skipping 'Sale Price vs. Year Built' scatter plot.")

        # 3. Overall quality vs. Sale price
        if "OverallQual" in df.columns and "SalePrice" in df.columns:
            overall_qual_pd = df.select(["OverallQual", "SalePrice", "Neighborhood"]).to_pandas()
            fig3 = px.scatter(overall_qual_pd, x="OverallQual", y="SalePrice",
                             color="Neighborhood",
                             trendline="ols",
                             title="Sale Price vs. Overall Quality",
                             labels={"OverallQual": "Overall Quality", "SalePrice": "Sale Price"},
                             hover_data=["Neighborhood"])
            scatter_plots["SalePrice_vs_OverallQual"] = fig3
            scatter_quality_path = self.output_dir / "sale_price_vs_overall_quality_scatter.html"
            fig3.write_html(str(scatter_quality_path))
            print(f"Scatter plot 'Sale Price vs. Overall Quality' saved to {scatter_quality_path}")
        else:
            print("Columns 'OverallQual' or 'SalePrice' not found. Skipping 'Sale Price vs. Overall Quality' scatter plot.")

        return scatter_plots

