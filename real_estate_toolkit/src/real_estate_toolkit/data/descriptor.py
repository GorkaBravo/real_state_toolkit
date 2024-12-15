from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union
from statistics import mean, median

@dataclass
class Descriptor:
    """Class for describing real estate data."""
    data: List[Dict[str, Any]]

    def none_ratio(self, columns: Union[List[str], str] = "all") -> Dict[str, float]:
        """Compute the ratio of None value per column."""
        if not self.data:
            return {}
        columns_to_check = (
            self.data[0].keys() if columns == "all" else columns
        )
        result = {}
        for col in columns_to_check:
            total = len(self.data)
            none_count = sum(1 for row in self.data if row.get(col) is None)
            result[col] = none_count / total
        return result

    def average(self, columns: Union[List[str], str] = "all") -> Dict[str, float]:
        """Compute the average value for numeric variables."""
        result = {}
        columns_to_check = (
            [col for col in self.data[0].keys() if isinstance(self.data[0][col], (int, float))]
            if columns == "all" else columns
        )
        for col in columns_to_check:
            values = [row[col] for row in self.data if isinstance(row.get(col), (int, float))]
            result[col] = mean(values) if values else 0
        return result

    def median(self, columns: Union[List[str], str] = "all") -> Dict[str, float]:
        """Compute the median value for numeric variables."""
        result = {}
        columns_to_check = (
            [col for col in self.data[0].keys() if isinstance(self.data[0][col], (int, float))]
            if columns == "all" else columns
        )
        for col in columns_to_check:
            values = [row[col] for row in self.data if isinstance(row.get(col), (int, float))]
            result[col] = median(values) if values else 0
        return result

    def percentile(self, columns: Union[List[str], str] = "all", percentile: int = 50) -> Dict[str, float]:
        """Compute the percentile value for numeric variables."""
        import numpy as np
        result = {}
        columns_to_check = (
            [col for col in self.data[0].keys() if isinstance(self.data[0][col], (int, float))]
            if columns == "all" else columns
        )
        for col in columns_to_check:
            values = [row[col] for row in self.data if isinstance(row.get(col), (int, float))]
            result[col] = np.percentile(values, percentile) if values else 0
        return result

    def type_and_mode(self, columns: Union[List[str], str] = "all") -> Dict[str, Union[Tuple[str, Any], Tuple[str, str]]]:
        """Compute the mode for variables."""
        from collections import Counter
        result = {}
        columns_to_check = (
            self.data[0].keys() if columns == "all" else columns
        )
        for col in columns_to_check:
            values = [row[col] for row in self.data if row.get(col) is not None]
            most_common = Counter(values).most_common(1)
            result[col] = (type(values[0]).__name__, most_common[0][0] if most_common else None)
        return result

import numpy as np
from typing import Dict, List, Union

class DescriptorNumpy:
    """Class for describing real estate data using NumPy."""
    
    def __init__(self, data: np.ndarray, columns: List[str] = None):
        if isinstance(data, list) and isinstance(data[0], dict):
            if columns is None:
                columns = list(data[0].keys())
            self.data = np.array([
                [row[col] if isinstance(row[col], (int, float)) else np.nan for col in columns]
                for row in data
            ])
        elif isinstance(data, np.ndarray):
            self.data = data
        else:
            raise ValueError("Data must be a NumPy array or a list of dictionaries")
        
        self.columns = columns or [f"col_{i}" for i in range(self.data.shape[1])]

    def none_ratio(self, columns: Union[List[str], str] = "all") -> Dict[str, float]:
        """Compute the ratio of None (NaN in NumPy) per column."""
        result = {}
        cols_to_check = self.columns if columns == "all" else columns
        for col in cols_to_check:
            col_index = self.columns.index(col)
            nan_count = np.sum(np.isnan(self.data[:, col_index]))
            result[col] = nan_count / self.data.shape[0]
        return result

    def average(self, columns: Union[List[str], str] = "all") -> Dict[str, float]:
        """Compute the average value for numeric variables."""
        result = {}
        cols_to_check = self.columns if columns == "all" else columns
        for col in cols_to_check:
            col_index = self.columns.index(col)
            numeric_values = self.data[:, col_index][~np.isnan(self.data[:, col_index])]
            result[col] = np.mean(numeric_values) if numeric_values.size else 0
        return result

    def median(self, columns: Union[List[str], str] = "all") -> Dict[str, float]:
        """Compute the median value for numeric variables."""
        result = {}
        cols_to_check = self.columns if columns == "all" else columns
        for col in cols_to_check:
            col_index = self.columns.index(col)
            numeric_values = self.data[:, col_index][~np.isnan(self.data[:, col_index])]
            result[col] = np.median(numeric_values) if numeric_values.size else 0
        return result

    def percentile(self, columns: Union[List[str], str] = "all", percentile: int = 50) -> Dict[str, float]:
        """Compute the percentile value for numeric variables."""
        result = {}
        cols_to_check = self.columns if columns == "all" else columns
        for col in cols_to_check:
            col_index = self.columns.index(col)
            numeric_values = self.data[:, col_index][~np.isnan(self.data[:, col_index])]
            result[col] = np.percentile(numeric_values, percentile) if numeric_values.size else 0
        return result

    def type_and_mode(self, columns: Union[List[str], str] = "all") -> Dict[str, Union[Tuple[str, Any], Tuple[str, str]]]:
        """
        Compute the type and mode for each column.
        Returns a dictionary where keys are column names and values are tuples of (type, mode).
        """
        from collections import Counter
        result = {}
        cols_to_check = self.columns if columns == "all" else columns
        for col in cols_to_check:
            col_index = self.columns.index(col)
            values = self.data[:, col_index]
            # Exclude NaN values
            non_nan_values = values[~np.isnan(values)] if values.dtype.kind in 'f' else values
            # Get the most common value (mode)
            if len(non_nan_values) > 0:
                most_common = Counter(non_nan_values).most_common(1)
                mode = most_common[0][0] if most_common else None
                result[col] = (type(non_nan_values[0]).__name__, mode)
            else:
                result[col] = ("NoneType", None)
        return result
