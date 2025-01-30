import warnings
warnings.filterwarnings("ignore")
import os
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
import numpy as np
import random
from model import test, EFF_KAN, REAL_KAN, ResNet
import pandas as pd

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

'''This code is directly copied from the flower tutorial, see https://github.com/adap/flower/blob/main/examples/flower-simulation-step-by-step-pytorch/Part-I/main.py'''

# A decorator for Hydra. This tells hydra to by default load the config in conf/base.yaml
@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    ## 1. Parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg))
    np.random.seed(cfg.seed)
    # torch.random.seed(cfg.seed)
    #torch.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    # Hydra automatically creates a directory for your experiments
    # by default it would be in <this directory>/outputs/<date>/<time>
    # you can retrieve the path to it as shown below. We'll use this path to
    # save the results of the simulation (see the last part of this main())
    save_path = HydraConfig.get().runtime.output_dir

    ## 2. Prepare your dataset
    # When simulating FL runs we have a lot of freedom on how the FL clients behave,
    # what data they have, how much data, etc. This is not possible in real FL settings.
    # In simulation you'd often encounter two types of dataset:
    #       * naturally partitioned, that come pre-partitioned by user id (e.g. FEMNIST,
    #         Shakespeare, SpeechCommands) and as a result these dataset have a fixed number
    #         of clients and a fixed amount/distribution of data for each client.
    #       * and others that are not partitioned in any way but are very popular in ML
    #         (e.g. MNIST, CIFAR-10/100). We can _synthetically_ partition these datasets
    #         into an arbitrary number of partitions and assign one to a different client.
    #         Synthetically partitioned dataset allow for simulating different data distribution
    #         scenarios to tests your ideas. The down side is that these might not reflect well
    #         the type of distributions encounter in the Wild.
    #
    # In this tutorial we are going to partition the MNIST dataset into 100 clients (the default
    # in our config -- but you can change this!) following a independent and identically distributed (IID)
    # sampling mechanism. This is arguably the simples way of partitioning data but it's a good fit
    # for this introductory tutorial.
    is_linear = True if cfg.model_type == 'REAL_KAN' or cfg.model_type == 'EFF_KAN' else False
    client_train_loaders, client_validation_loaders, global_valid_loader, _ = prepare_dataset(
        cfg.num_clients, cfg.batch_size, linear=is_linear, equal_distribution=False
    )
    exit()
    
    '''net = ConvNeXtKAN_v1()
    client_train = client_train_loaders[0]
    client_valid = client_validation_loaders[0]
    epochs = 3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = Adam(net.parameters(), lr=0.0001, weight_decay=1e-5)
    train(net, client_train, optimizer, 3, device)
    loss, accuracy, f1, precision, recall = test(net, client_valid, device)
    print(f'Validation Loss: {loss}, accuracy: {accuracy}, f1: {f1}, precision: {precision}, recall: {recall}')
    exit() 
    '''
    ## 3. Define your clients
    # Unlike in standard FL (e.g. see the quickstart-pytorch or quickstart-tensorflow examples in the Flower repo),
    # in simulation we don't want to manually launch clients. We delegate that to the VirtualClientEngine.
    # What we need to provide to start_simulation() with is a function that can be called at any point in time to
    # create a client. This is what the line below exactly returns.
    client_fn = generate_client_fn(client_train_loaders, client_validation_loaders, cfg.num_classes, cfg)

    ## 4. Define your strategy
    # A flower strategy orchestrates your FL pipeline. Although it is present in all stages of the FL process
    # each strategy often differs from others depending on how the model _aggregation_ is performed. This happens
    # in the strategy's `aggregate_fit()` method. In this tutorial we choose FedAvg, which simply takes the average
    # of the models received from the clients that participated in a FL round doing fit().
    # You can implement a custom strategy to have full control on all aspects including: how the clients are sampled,
    # how updated models from the clients are aggregated, how the model is evaluated on the server, etc
    # To control how many clients are sampled, strategies often use a combination of two parameters `fraction_{}` and `min_{}_clients`
    # where `{}` can be either `fit` or `evaluate`, depending on the FL stage. The final number of clients sampled is given by the formula
    # ``` # an equivalent bit of code is used by the strategies' num_fit_clients() and num_evaluate_clients() built-in methods.
    #         num_clients = int(num_available_clients * self.fraction_fit)
    #         clients_to_do_fit = max(num_clients, self.min_fit_clients)
    # ```
    

    strategy_type = string_to_class(cfg.strategy_config.module_name, cfg.strategy_config.class_name)
    strategy = strategy_type(
        proximal_mu = 0.5,
        fraction_fit=0.5,  # in simulation, since all clients are available at all times, we can just use `min_fit_clients` to control exactly how many clients we want to involve during fit
        min_fit_clients=cfg.num_clients_per_round_fit,  # number of clients to sample for fit()
        fraction_evaluate=0.5,  # similar to fraction_fit, we don't need to use this argument.
        min_evaluate_clients=cfg.num_clients_per_round_eval,  # number of clients to sample for evaluate()
        min_available_clients=cfg.num_clients,  # total clients in the simulation
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=get_on_fit_config(
            cfg.config_fit
        ),  # a function to execute to obtain the configuration to send to the clients during fit()
        evaluate_fn=get_evaluate_fn(cfg.num_classes, global_valid_loader, cfg),
    )  # a function to run on the server side to evaluate the global model.

    ## 5. Start Simulation
    # With the dataset partitioned, the client function and the strategy ready, we can now launch the simulation!
    history = fl.simulation.start_simulation(
        client_fn=client_fn,  # a function that spawns a particular client
        num_clients=cfg.num_clients,  # total number of clients
        config=fl.server.ServerConfig(
            num_rounds=cfg.num_rounds
        ),  # minimal config for the server loop telling the number of rounds in FL
        strategy=strategy,  # our strategy of choice
        client_resources={
            "num_cpus": 10,
            "num_gpus": 0,
        },  # (optional) controls the degree of parallelism of your simulation.
        # Lower resources per client allow for more clients to run concurrently
        # (but need to be set taking into account the compute/memory footprint of your run)
        # `num_cpus` is an absolute number (integer) indicating the number of threads a client should be allocated
        # `num_gpus` is a ratio indicating the portion of gpu memory that a client needs.
    )
    #print(history)

    # ^ Following the above comment about `client_resources`. if you set `num_gpus` to 0.5 and you have one GPU in your system,
    # then your simulation would run 2 clients concurrently. If in your round you have more than 2 clients, then clients will wait
    # until resources are available from them. This scheduling is done under-the-hood for you so you don't have to worry about it.
    # What is really important is that you set your `num_gpus` value correctly for the task your clients do. For example, if you are training
    # a large model, then you'll likely see `nvidia-smi` reporting a large memory usage of you clients. In those settings, you might need to
    # leave `num_gpus` as a high value (0.5 or even 1.0). For smaller models, like the one in this tutorial, your GPU would likely be capable
    # of running at least 2 or more (depending on your GPU model.)
    # Please note that GPU memory is only one dimension to consider when optimising your simulation. Other aspects such as compute footprint
    # and I/O to the filesystem or data preprocessing might affect your simulation  (and tweaking `num_gpus` would not translate into speedups)
    # Finally, please note that these gpu limits are not enforced, meaning that a client can still go beyond the limit initially assigned, if
    # this happens, your might get some out-of-memory (OOM) errors.
    # Log simulation metrics to TensorBoard and DataFrames
    centralized_metrics_list = []
    distributed_metrics_list = []

    for metric_name, metric_values in history.metrics_centralized.items():
        for round_number, (round_index, metric_value) in enumerate(metric_values):
            # writer.add_scalar(f"Metrics/Centralized/{metric_name}", metric_value, round_index)
            centralized_metrics_list.append({
                "Round": round_index,
                "Metric": metric_name,
                "Value": metric_value
            })

    for metric_name, metric_values in history.metrics_distributed.items():
        for round_number, (round_index, metric_value) in enumerate(metric_values):
            # writer.add_scalar(f"Metrics/Distributed/{metric_name}", metric_value, round_index)
            distributed_metrics_list.append({
                "Round": round_index,
                "Metric": metric_name,
                "Value": metric_value
            })

    # Convert to DataFrame after accumulating all data
    centralized_metrics_df = pd.DataFrame(centralized_metrics_list)
    distributed_metrics_df = pd.DataFrame(distributed_metrics_list)

    # Save metrics DataFrames to CSV files
    centralized_metrics_path = Path(save_path) / f"{str(cfg.model_type)}_{str(cfg.strategy_config.class_name)}_centralized_metrics.csv"
    distributed_metrics_path = Path(save_path) / f"{str(cfg.model_type)}_{str(cfg.strategy_config.class_name)}_distributed_metrics.csv"

    centralized_metrics_df.to_csv(centralized_metrics_path, index=False)
    distributed_metrics_df.to_csv(distributed_metrics_path, index=False)
    ## 6. Save your results
    # (This is one way of saving results, others are of course valid :) )
    # Now that the simulation is completed, we could save the results into the directory
    # that Hydra created automatically at the beginning of the experiment.
    results_path = Path(save_path) / "results.pkl"

    # add the history returned by the strategy into a standard Python dictionary
    # you can add more content if you wish (note that in the directory created by
    # Hydra, you'll already have the config used as well as the log)
    results = {"history": history, "anythingelse": "here"}

    # save the results as a python pickle
    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)


def test_models():
    models = [[ResNet(num_classes=2, softmax=False), 'ResNet'], [EFF_KAN([224 * 224, 224, 128, 2]), 'EFF_KAN'], [REAL_KAN([224 * 224, 224, 128, 2]), 'REAL_KAN']]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.dirname(os.path.realpath(__file__))

    for model, model_type in models:
        PATH = f'{model_path}/model_dicts/{model_type}.pth'
        is_linear = False if model_type == 'ResNet' else True
        _, _, _, global_test_loader = prepare_dataset(
            3, 32, linear=is_linear)
        model.load_state_dict(torch.load(PATH, weights_only=True))

        loss, f1, precision, recall = test(model, global_test_loader, device)
        print(f'Testing {model_type}')
        print(f'Testloss: {loss}, Test F1-Score: {f1}, Test Precision: {precision}, Test Recall: {recall}')


if __name__ == "__main__":
    main()
    # test_models()

