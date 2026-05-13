"""
Abstract base class for federated learning strategies.

All strategies (H-SFP, Serverless, etc.) must implement this interface
to ensure consistent usage from the main training script.
"""

from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """
    Abstract base class defining the interface for FL strategies.

    Subclasses implement specific training pipelines while sharing
    the same entry points for building, training, and evaluation.

    To add a new strategy:
        1. Create a new file in strategies/ (e.g., serverless.py)
        2. Subclass BaseStrategy
        3. Implement all abstract methods
        4. Register it in strategies/__init__.py
    """

    def __init__(self, config, client_model, edge_model, cloud_model, test_dataset=None):
        self.config = config
        self.client_model = client_model
        self.edge_model = edge_model
        self.cloud_model = cloud_model
        self.test_dataset = test_dataset

    @abstractmethod
    def build_hierarchy(self):
        """Build the hierarchical topology (clients → edges → cloud)."""
        ...

    @abstractmethod
    def train(self, train_dataset, valid_dataset, test_dataset, user_groups, epochs):
        """
        Run the full training loop.

        Args:
            train_dataset: Training dataset.
            valid_dataset: Validation dataset.
            test_dataset: Test dataset.
            user_groups: Dict mapping user_id → data indices.
            epochs: Number of global rounds.

        Returns:
            Dict with training results and metrics.
        """
        ...

    @abstractmethod
    def print_structure(self):
        """Print the topology of the hierarchy."""
        ...
