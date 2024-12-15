from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import csv

@dataclass
class DataLoader:
    """Class for loading and basic processing of real estate data."""
    data_path: Path
    
    def load_data_from_csv(self) -> List[Dict[str, Any]]:
        """Load data from CSV file into a list of dictionaries."""
        try:
            with open(self.data_path, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                return [row for row in reader]
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found at {self.data_path}")
        except Exception as e:
            raise RuntimeError(f"Error reading file: {str(e)}")
    
    def validate_columns(self, required_columns: List[str]) -> bool:
        """Validate that all required columns are present in the dataset."""
        data = self.load_data_from_csv()
        if not data:
            return False
        actual_columns = set(data[0].keys())
        return all(column in actual_columns for column in required_columns)
