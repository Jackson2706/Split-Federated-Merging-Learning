from abc import ABC, abstractmethod


class Aggregator(ABC):
    def __init__(self, args):
        self.args = args

    @abstractmethod
    def aggregate(self, client_updates):
        """
        client_updates: dữ liệu cập nhật từ các client,
        có thể là list of state_dict hoặc list of tuples tuỳ thuật toán
        """
        pass