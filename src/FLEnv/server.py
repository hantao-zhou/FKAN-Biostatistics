
from collections import OrderedDict

import torch
from omegaconf import DictConfig

from model import Dummy_Model, test, ConvNeXtKAN_v1
import numpy as np
from flwr.common import Metrics
from typing import List, Tuple

'''This code is directly copied from the flower tutorial, see https://github.com/adap/flower/blob/main/examples/flower-simulation-step-by-step-pytorch/Part-I/server.py'''

def get_on_fit_config(config: DictConfig):
    """Return function that prepares config to send to clients."""

    def fit_config_fn(server_round: int):
        # This function will be executed by the strategy in its
        # `configure_fit()` method.

        # Here we are returning the same config on each round but
        # here you might use the `server_round` input argument to
        # adapt over time these settings so clients. For example, you
        # might want clients to use a different learning rate at later
        # stages in the FL process (e.g. smaller lr after N rounds)

        return {
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "local_epochs": config.local_epochs,
        }

    return fit_config_fn


def get_evaluate_fn(num_classes: int, testloader):
    """Define function for global evaluation on the server."""

    def evaluate_fn(server_round: int, parameters, config):
        # This function is called by the strategy's `evaluate()` method
        # and receives as input arguments the current round number and the
        # parameters of the global model.
        # this function takes these parameters and evaluates the global model
        # on a evaluation / test dataset.

        model = ConvNeXtKAN_v1(num_classes=num_classes)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        # Here we evaluate the global model on the test set. Recall that in more
        # realistic settings you'd only do this at the end of your FL experiment
        # you can use the `server_round` input argument to determine if this is the
        # last round. If it's not, then preferably use a global validation set.
        loss, accuracy, f1, precision, recall = test(model, testloader, device)

        # Report the loss and any other metric (inside a dictionary). In this case
        # we report the global test accuracy.
        return loss, {"Accuracy": round(accuracy, 3), "F1": round(f1, 3), "Precision": round(precision, 3), "Recall": round(recall, 3)}

    return evaluate_fn


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    examples = round(sum([num_examples for num_examples, _ in metrics]), 3)
    accuracie = round(sum([num_examples * m["accuracy"] for num_examples, m in metrics]) / examples, 3)
    recall = round(sum([num_examples * m["recall"] for num_examples, m in metrics]) / examples, 3)
    precision = round(sum([num_examples * m["precision"] for num_examples, m in metrics]) / examples, 3)
    f1 = round(sum([num_examples * m["f1"] for num_examples, m in metrics]) / examples, 3)
    

    print(f'----------')
    print(f'averaged metrics for all clients:')
    print(f'Accuracy: {accuracie}, F1: {f1}, Precision: {precision}, Recall: {recall}')
    print(f'----------')

    # Aggregate and return custom metric (weighted average)
    return {"Accuracy": accuracie, "Recall": recall, "Precision": precision, "F1": f1}