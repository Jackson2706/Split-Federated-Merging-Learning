import pickle
from pathlib import Path

import flwr as fl
import hydra
from client import generate_client_fn
from dataset import prepare_dataset
from fednova import FedNova
from flwr.common import ndarrays_to_parameters
from flwr.server.strategy import FedProx
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from server import get_evaluate_fn, get_in_fit_config


@hydra.main(config_path = "config", config_name="base", version_base=None)
def main(cfg: DictConfig):
    ## 1. Parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg))

    ## 2. Prepare the dataset
    trainloaders, validationloaders, testloader = prepare_dataset(
        cfg.num_clients,cfg.batch_size, cfg.val_ratio
    )

    # 3. Define the Flower client
    client_fn = generate_client_fn(
        trainloaders, validationloaders, cfg.num_classes
    )

    # # 4. Define the strategu
    # strategy = fl.server.strategy.FedAvg(
    #     fraction_fit=0.5,
    #     min_fit_clients=cfg.num_clients_per_round_fit,
    #     fraction_evaluate=0.5,
    #     min_evaluate_clients=cfg.num_clients_per_round_eval,
    #     min_available_clients=cfg.num_clients,
    #     on_fit_config_fn=get_in_fit_config(cfg.config_fit),
    #     evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),
    # )
    strategy = FedProx(
        fraction_fit=0.5,
        min_fit_clients=cfg.num_clients_per_round_fit,
        fraction_evaluate=0.5,
        min_evaluate_clients=cfg.num_clients_per_round_eval,
        min_available_clients=cfg.num_clients,
        on_fit_config_fn=get_in_fit_config(cfg.config_fit),
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),
        proximal_mu=0.01
    )

    strategy = fl.server.strategy.fed
    ### 5. Start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config = fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
    )

    ### 6. Save the results
    save_path = HydraConfig.get().runtime.output_dir
    result_path = Path(save_path) / "results.pkl"

    results = {
        "history": history,
        "config": cfg,
    }
    with open(str(result_path), "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()