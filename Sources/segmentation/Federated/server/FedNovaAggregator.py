import copy
import sys

from .Aggregator import Aggregator


class FedNovaAggregator(Aggregator):
    def aggregate(self, client_updates, global_weights, local_weights):
        """
        Args:
            client_updates: list of tuples (delta_weights, local_steps)
            global_weights: current global model weights (state_dict)
        
        Returns:
            new_global_weights: updated global model weights after aggregation
            server_comm: communication overhead the server has to handle
        """
        total_steps = sum([steps for _, steps in client_updates])
        if total_steps == 0:
            raise ValueError("Sum of local steps is zero, cannot normalize")
        
        server_comm = sys.getsizeof(client_updates) + sys.getsizeof(global_weights)

        agg_delta = None
        for delta_weights, steps in client_updates:
            weighted_delta = {k: v * steps for k, v in delta_weights.items()}
            if agg_delta is None:
                agg_delta = copy.deepcopy(weighted_delta)
            else:
                for k in agg_delta.keys():
                    agg_delta[k] += weighted_delta[k]

        # Normalize aggregated delta by total steps
        for k in agg_delta.keys():
            agg_delta[k] /= total_steps

        # Update global weights
        new_global_weights = copy.deepcopy(global_weights)
        for k in new_global_weights.keys():
            new_global_weights[k] = new_global_weights[k].float() + agg_delta[k]


        # Return the updated global weights and the communication overhead
        return new_global_weights, server_comm