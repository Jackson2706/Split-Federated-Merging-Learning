from .FedAvgAggregator import FedAvgAggregator


class FedSGDAggregator(FedAvgAggregator):
    # FedSGD can inherit FedAvgAggregator since the aggregation is identical
    pass