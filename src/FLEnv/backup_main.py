import pickle
from pathlib import Path
import importlib
import flwr as fl
import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import logging
from client import generate_client_fn
from dataset import prepare_dataset
from server import get_evaluate_fn, get_on_fit_config, weighted_average
from model import Dummy_Model, train, test, ConvNeXtKAN_v1
from torch.optim import SGD, Adam
import numpy as np
import random
import pandas as pd
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard writer

def string_to_class(module_name, class_name):
    try:
        module = importlib.import_module(module_name)
        try:
            class_ = getattr(module, class_name)
        except AttributeError:
            logging.error('Class does not exist')
    except ImportError:
        logging.error('Module does not exist')
    return class_ or None

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir="tensorboard_logs")  # Log directory for TensorBoard

    # 1. Parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg))
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    save_path = HydraConfig.get().runtime.output_dir
    # import ipdb; ipdb.set_trace()
    # 2. Prepare your dataset
    client_train_loaders, client_validation_loaders, global_valid_loader, global_test_loader = prepare_dataset(
        cfg.num_clients, cfg.batch_size
    )

    # 3. Define your clients
    client_fn = generate_client_fn(
        client_train_loaders, client_validation_loaders, cfg.num_classes
    )

    # 4. Define your strategy
    strategy_type = string_to_class(cfg.strategy_config.module_name, cfg.strategy_config.class_name)
    strategy = strategy_type(
        min_fit_clients=cfg.num_clients_per_round_fit,
        min_evaluate_clients=cfg.num_clients_per_round_eval,
        min_available_clients=cfg.num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=get_on_fit_config(cfg.config_fit),
        evaluate_fn=get_evaluate_fn(cfg.num_classes, global_valid_loader),
    )

    # 5. Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": cfg.client_resources.num_cpus, "num_gpus": cfg.client_resources.num_gpus},
    )

    # Initialize DataFrames for metrics
    centralized_metrics_df = pd.DataFrame()
    distributed_metrics_df = pd.DataFrame()

    # Log simulation metrics to TensorBoard and DataFrames
    centralized_metrics_list = []
    distributed_metrics_list = []

    for metric_name, metric_values in history.metrics_centralized.items():
        for round_number, (round_index, metric_value) in enumerate(metric_values):
            writer.add_scalar(f"Metrics/Centralized/{metric_name}", metric_value, round_index)
            centralized_metrics_list.append({
                "Round": round_index,
                "Metric": metric_name,
                "Value": metric_value
            })

    for metric_name, metric_values in history.metrics_distributed.items():
        for round_number, (round_index, metric_value) in enumerate(metric_values):
            writer.add_scalar(f"Metrics/Distributed/{metric_name}", metric_value, round_index)
            distributed_metrics_list.append({
                "Round": round_index,
                "Metric": metric_name,
                "Value": metric_value
            })

    # Convert to DataFrame after accumulating all data
    centralized_metrics_df = pd.DataFrame(centralized_metrics_list)
    distributed_metrics_df = pd.DataFrame(distributed_metrics_list)

    # Save metrics DataFrames to CSV files
    centralized_metrics_path = Path(save_path) / "centralized_metrics.csv"
    distributed_metrics_path = Path(save_path) / "distributed_metrics.csv"

    centralized_metrics_df.to_csv(centralized_metrics_path, index=False)
    distributed_metrics_df.to_csv(distributed_metrics_path, index=False)


    # Log final results
    writer.add_hparams(
        {
            "num_rounds": cfg.num_rounds,
            "num_clients": cfg.num_clients,
            "batch_size": cfg.batch_size,
            "num_classes": cfg.num_classes
        },
        {
            "final_accuracy": history.metrics_centralized["Accuracy"][-1][1]
        },
    )

    writer.close()  # Close the TensorBoard writer

    # 6. Save your results
    results_path = Path(save_path) / "results.pkl"
    results = {"history": history, "anythingelse": "here"}

    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
