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
        num_clients = len(client_updates)
        total_steps = sum([steps for _, steps in client_updates])
        if total_steps == 0:
            raise ValueError("Sum of local steps is zero, cannot normalize")

        server_comm = sys.getsizeof(client_updates) + sys.getsizeof(global_weights)

        # FedNova (Wang et al., 2020) with equal client weights p_i = 1/m
        # (consistent with the unweighted FedAvg used elsewhere in this repo).
        # Each client returns its *normalized* update delta_i = (w_i - w_t)/tau_i.
        # The global step is:
        #     w_{t+1} = w_t + tau_eff * mean_i(delta_i),   tau_eff = mean_i(tau_i)
        # This correctly reduces to FedAvg when all tau_i are equal; the previous
        # implementation (sum(delta_i * tau_i)/total_steps) shrank the step by ~tau.
        tau_eff = total_steps / num_clients

        agg_delta = None
        for delta_weights, _ in client_updates:
            if agg_delta is None:
                agg_delta = copy.deepcopy(delta_weights)
            else:
                for k in agg_delta.keys():
                    agg_delta[k] += delta_weights[k]

        # mean over clients, then scale by the effective number of local steps
        for k in agg_delta.keys():
            agg_delta[k] = agg_delta[k] / num_clients * tau_eff

        # Update global weights
        new_global_weights = copy.deepcopy(global_weights)
        for k in new_global_weights.keys():
            new_global_weights[k] = new_global_weights[k].float() + agg_delta[k]

        # Return the updated global weights and the communication overhead
        return new_global_weights, server_comm