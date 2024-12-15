from random import choice, gauss, randint
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional
from agent_based_model.houses import House
from agent_based_model.house_market import HousingMarket

class Segment(Enum):
    FANCY = "FANCY"
    OPTIMIZER = "OPTIMIZER"
    AVERAGE = "AVERAGE"


@dataclass
class Consumer:
    id: int
    annual_income: float
    children_number: int
    segment: Segment
    house: Optional[House] = None
    savings: float = 0.0
    saving_rate: float = 0.3
    interest_rate: float = 0.05
    has_house: bool = False

    def compute_savings(self, years: int) -> None:
        """
        Compute the future value of savings after a given number of years.
        This includes the growth of current savings plus the growth of annual contributions.
        """
        if self.interest_rate <= 0:
            raise ValueError("Interest rate must be greater than zero.")

        annual_savings = self.annual_income * self.saving_rate
        # Future value of the initial savings after 'years' years
        initial_growth = self.savings * (1 + self.interest_rate) ** years
        # Future value of annual contributions (annuity)
        annual_growth = annual_savings * (((1 + self.interest_rate) ** years - 1) / self.interest_rate)

        self.savings = initial_growth + annual_growth

    def buy_a_house(self, housing_market: HousingMarket) -> None:
        """
        Attempt to buy a house that meets the consumer's financial and segment requirements.
        """
        # Pass the current savings and segment to the housing market
        candidates = housing_market.get_houses_that_meet_requirements(
            budget=self.savings,
            segment=self.segment
        )

        if not candidates:
            # No suitable house found
            return

        # Choose the first suitable house (or apply more complex selection logic if needed)
        house_to_buy = candidates[0]

        # Mark the chosen house as sold and adjust the consumer's savings accordingly
        house_to_buy.sell_house()
        self.house = house_to_buy
        self.savings -= house_to_buy.price
        self.has_house = True
