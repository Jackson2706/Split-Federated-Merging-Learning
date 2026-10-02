from abc import ABC, abstractmethod


class Aggregator(ABC):
    def __init__(self, args):
        self.args = args

    @abstractmethod
    def aggregate(self, client_updates):
        """
        client_updates: update data from the clients,
        may be a list of state_dicts or list of tuples depending on the algorithm
        """
        pass