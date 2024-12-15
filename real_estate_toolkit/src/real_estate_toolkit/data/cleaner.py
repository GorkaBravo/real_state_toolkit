from dataclasses import dataclass
from typing import Dict, List, Any
import re

@dataclass
class Cleaner:
    """Class for cleaning real estate data."""
    data: List[Dict[str, Any]]

    def rename_with_best_practices(self) -> None:
        """
        Rename the columns with best practices (e.g., snake_case and descriptive names).
        Converts column names to snake_case and stores the updated data in place.
        """
        def to_snake_case(name: str) -> str:
            """Convert a string to snake_case."""
            return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
        
        # Rename keys in all rows
        if self.data:
            self.data = [
                {to_snake_case(key): value for key, value in row.items()}
                for row in self.data
            ]

    def na_to_none(self) -> List[Dict[str, Any]]:
        """
        Replace NA to None in all values with NA in the dictionary.
        NA can include: None, 'NA', 'N/A', 'na', 'n/a', empty strings, or numpy.nan.
        """
        def clean_value(value: Any) -> Any:
            """Standardize 'NA' values to None."""
            if value in [None, "NA", "N/A", "na", "n/a", ""]:
                return None
            try:
                import numpy as np
                if isinstance(value, float) and np.isnan(value):
                    return None
            except ImportError:
                pass
            return value
        
        # Replace 'NA' values in all rows
        return [
            {key: clean_value(value) for key, value in row.items()}
            for row in self.data
        ]
