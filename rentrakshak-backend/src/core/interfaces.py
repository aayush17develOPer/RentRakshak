# src/core/interfaces.py
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

class IRiskAnalyzer(ABC):
    """
    Open/Closed Principle: All risk modules (Vision, Price, Text, and future Legal) 
    must implement this interface. 
    """
    @abstractmethod
    async def analyze(self, listing_data: Dict[str, Any]) -> Tuple[float, str]:
        """
        Analyzes the listing data.
        Returns:
            Tuple[float, str]: A tuple containing the risk score (0-100) and the explanation proof.
        """
        pass

class IListingScraper(ABC):
    """
    Single Responsibility Principle: Dedicated strictly to fetching data.
    """
    @abstractmethod
    async def extract_listing(self, url: str) -> Dict[str, Any]:
        """
        Extracts raw data from a given property URL.
        """
        pass