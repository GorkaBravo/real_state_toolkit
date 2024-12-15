from enum import Enum, auto
from dataclasses import dataclass
import random
from random import gauss, randint, shuffle
from typing import List, Dict, Any
from .houses import House
from .house_market import HousingMarket
from .consumers import Segment, Consumer

class CleaningMarketMechanism(Enum):
    INCOME_ORDER_DESCENDANT = auto()
    INCOME_ORDER_ASCENDANT = auto()
    RANDOM = auto()

@dataclass
class AnnualIncomeStatistics:
    minimum: float
    average: float
    standard_deviation: float
    maximum: float

@dataclass
class ChildrenRange:
    minimum: int = 0
    maximum: int = 5

@dataclass
class Simulation:
    housing_market_data: List[Dict[str, Any]]
    consumers_number: int
    years: int
    annual_income: AnnualIncomeStatistics
    children_range: ChildrenRange
    cleaning_market_mechanism: CleaningMarketMechanism
    down_payment_percentage: float = 0.2
    saving_rate: float = 0.3
    interest_rate: float = 0.05

    def __post_init__(self):
 
        self.consumers = []  # Lista de objetos Consumer

    def create_housing_market(self):
        """
        Initialize market with houses.
        Converts raw housing data into House objects and validates the data.
        Assigns the housing market to the class.
        """
        # Lista de claves permitidas según el constructor de House
        allowed_keys = {"id", "price", "area", "bedrooms", "year_built", "quality_score", "available"}

        houses = []  # Lista para almacenar las instancias de House

        for house_data in self.housing_market_data:
            # Transformar los datos para incluir las claves requeridas por House
            try:
                transformed_data = {
                    "id": int(house_data.get("id", 0)),  # Asegurar que siempre hay un ID
                    "price": float(house_data.get("sale_price", 0)),  # Mapear 'sale_price' a 'price'
                    "area": float(house_data.get("gr_liv_area", 0)),  # Mapear 'gr_liv_area' a 'area'
                    "bedrooms": int(house_data.get("bedroom_abv_gr", 0)),  # Mapear 'bedroom_abv_gr' a 'bedrooms'
                    "year_built": int(house_data.get("year_built", 0)),  # Tomar el año de construcción
                    "quality_score": None,  # Inicialmente no proporcionamos calidad (puede calcularse después)
                    "available": True  # Suponemos que todas las casas están disponibles inicialmente
             }

            # Crear la instancia de House con los datos transformados
                house = House(**transformed_data)
                houses.append(house)

            except TypeError as e:
                raise ValueError(f"Error al crear la casa con los datos: {house_data}. Detalle del error: {e}")

    # Inicializar el mercado de viviendas con la lista de casas
        self.housing_market = HousingMarket(houses)




    def create_consumers(self) -> None:
        """
        Generate consumer population.
        """
        for _ in range(self.consumers_number):
            while True:
                income = gauss(self.annual_income.average, self.annual_income.standard_deviation)
                if self.annual_income.minimum <= income <= self.annual_income.maximum:
                    break

            children = randint(self.children_range.minimum, self.children_range.maximum)
            segment = random.choice(list(Segment))  # Selecciona un segmento al azar

            consumer = Consumer(
                id=len(self.consumers),
                annual_income=income,
                children_number=children,
                segment=segment,
                saving_rate=self.saving_rate,
                interest_rate=self.interest_rate
            )

            self.consumers.append(consumer)

    def compute_consumers_savings(self) -> None:
        """
        Calculate savings for all consumers.
        """
        for consumer in self.consumers:
            consumer.savings = consumer.annual_income * self.saving_rate

    def clean_the_market(self) -> None:
    
        # Ordenar consumidores según el mecanismo seleccionado
        if self.cleaning_market_mechanism == CleaningMarketMechanism.INCOME_ORDER_DESCENDANT:
            self.consumers.sort(key=lambda c: c.annual_income, reverse=True)
        elif self.cleaning_market_mechanism == CleaningMarketMechanism.INCOME_ORDER_ASCENDANT:
            self.consumers.sort(key=lambda c: c.annual_income)
        elif self.cleaning_market_mechanism == CleaningMarketMechanism.RANDOM:
            shuffle(self.consumers)

    # Intentar realizar la compra para cada consumidor
        for consumer in self.consumers:
            if not consumer.has_house:  # Solo intentan comprar quienes no tienen casa
                success = self.housing_market.buy_a_house(
                    consumer,
                    down_payment_percentage=self.down_payment_percentage,
                    interest_rate=self.interest_rate
                )
                if success:
                    consumer.has_house = True  # Actualiza si la compra fue exitosa


    def compute_owners_population_rate(self) -> float:
        """
        Compute the owners population rate after the market is clean.
        """
        owners = sum(1 for consumer in self.consumers if consumer.has_house)
        return owners / len(self.consumers)

    def compute_houses_availability_rate(self) -> float:
        """
        Compute the houses availability rate after the market is clean.
        """
        available_houses = sum(1 for house in self.housing_market.houses if not house.sell_house)
        total_houses = len(self.housing_market.houses)
        return available_houses / total_houses