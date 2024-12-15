from enum import Enum
from dataclasses import dataclass
from typing import Optional

class QualityScore(Enum):
    EXCELLENT = 5
    GOOD = 4
    AVERAGE = 3
    FAIR = 2
    POOR = 1

@dataclass
class House:
    id: int
    price: float
    area: float
    bedrooms: int
    year_built: int
    quality_score: Optional[QualityScore]
    available: bool = True

    def calculate_price_per_square_foot(self) -> float:
        """
        Calculate and return the price per square foot.
        """
        if self.area <= 0:
            raise ValueError("Area must be greater than zero to calculate price per square foot.")
        return round(self.price / self.area, 2)

    def is_new_construction(self, current_year: int = 2024) -> bool:
        """
        Determine if house is considered new construction (< 5 years old).
        """
        return (current_year - self.year_built) < 5

    def get_quality_score(self) -> None:
        """
        Generate a quality score based on house attributes.
        """
        if self.quality_score is None:
            age = 2024 - self.year_built
            if age <= 5:
                self.quality_score = QualityScore.EXCELLENT
            elif age <= 15:
                self.quality_score = QualityScore.GOOD
            elif age <= 30:
                self.quality_score = QualityScore.AVERAGE
            elif age <= 50:
                self.quality_score = QualityScore.FAIR
            else:
                self.quality_score = QualityScore.POOR

    def sell_house(self) -> None:
        """
        Mark house as sold.
        """
        if not self.available:
            raise ValueError("House is already sold.")
        self.available = False
