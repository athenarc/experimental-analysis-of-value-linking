from abc import ABC, abstractmethod
from typing import List


class FilterABC(ABC):
    """
    Abstract base class for filtering results after querying an index
    using a coarse-to-fine approach.

    Implementations of this class should provide the ability to add keyword-value pairs
    and later filter these pairs based on custom criteria.
    """

    @abstractmethod
    def add_pair(self, keyword: str, value_pair: tuple):
        """
        Add a pair consisting of a keyword and a value pair.

        Parameters:
            keyword (str): The keyword to associate with the value pair.
            value_pair (tuple): A tuple of two elements where the first element is the value
                                and the second element is the formatted value.
        """
        pass

    def filter(self) -> List[str]:
        """
        Filter the added pairs and return a list of formatted values that meet the filtering criteria.

        Returns:
            List[str]: A list of filtered formatted values.
        """
        pass
