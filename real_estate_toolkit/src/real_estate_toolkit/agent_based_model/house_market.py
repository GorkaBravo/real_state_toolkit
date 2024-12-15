from typing import List, Optional
from .houses import House

class HousingMarket:
    def __init__(self, houses: List[House]):
        # Validate input to ensure no None values in the houses list
        if not houses:
            raise ValueError("Houses list is empty or None.")
        if any(house is None for house in houses):
            raise ValueError("Houses list contains None values.")
        
        self.houses: List[House] = houses

    def get_house_by_id(self, house_id: int) -> Optional[House]:
        """
        Retrieve specific house by ID.
        """
        for house in self.houses:
            if house.id == house_id:
                return house
        return None  # Return None if house with the given ID is not found

    def calculate_average_price(self, bedrooms: Optional[int] = None) -> float:
        """
        Calculate average house price, optionally filtered by bedrooms.
        """
        # Filter houses by number of bedrooms if specified
        filtered_houses = [house.price for house in self.houses if bedrooms is None or house.bedrooms == bedrooms]
        if not filtered_houses:
            return 0.0  # Handle empty list gracefully
        return sum(filtered_houses) / len(filtered_houses)

    def get_houses_that_meet_requirements(self, max_price: int, segment: str) -> List[House]:
        """
        Filter houses based on buyer requirements.
        """
        result = []
        for house in self.houses:
            if house.price is not None and house.price <= max_price:
                if segment == "FANCY" and not house.is_new_construction():
                    continue
                result.append(house)
        if not result:
            print(f"Warning: No houses meet the requirements for segment {segment} and max price {max_price}.")
        return result
