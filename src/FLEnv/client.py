from collections import OrderedDict
from typing import Dict, Tuple

import flwr as fl
import torch
from flwr.common import NDArrays, Scalar
import numpy as np
from model import test, train, REAL_KAN, EFF_KAN, ResNet

'''This code is directly copied from the flower tutorial, see https://github.com/adap/flower/blob/main/examples/flower-simulation-step-by-step-pytorch/Part-I/client.py'''

class FlowerClient(fl.client.NumPyClient):
    """Define a Flower Client."""

    def __init__(self, trainloader, valloader, num_classes, config) -> None:
        super().__init__()

        # the dataloaders that point to the data associated to this client
        self.trainloader = trainloader
        self.valloader = valloader

        # a model that is randomly initialised at first
        self.model = None
        if config.model_type == 'REAL_KAN':
            self.model = REAL_KAN([224 * 224, 224, 128, num_classes])
        elif config.model_type == 'EFF_KAN':
            self.model = EFF_KAN([224 * 224, 224, 128, num_classes])
        else: self.model = ResNet(num_classes=num_classes, softmax=False)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.config_fit.lr, weight_decay=config.config_fit.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.8)

        # figure out if this client has access to GPU support or not
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("mps")

    def set_parameters(self, parameters):
        """Receive parameters and apply them to the local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)

        state_dict = state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})

        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract model parameters and return them as a list of numpy arrays."""

        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """Train model received by the server (parameters) using the data.

        that belongs to this client. Then, send it back to the server.
        """

        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)

        # fetch elements in the config sent by the server. Note that having a config
        # sent by the server each time a client needs to participate is a simple but
        # powerful mechanism to adjust these hyperparameters during the FL process. For
        # example, maybe you want clients to reduce their LR after a number of FL rounds.
        # or you want clients to do more local epochs at later stages in the simulation
        # you can control these by customising what you pass to `on_fit_config_fn` when
        # defining your strategy.
        lr = config["lr"]
        weight_decay = config["weight_decay"]
        epochs = config["local_epochs"]

        # a very standard looking optimiser
        
        # do local training. This function is identical to what you might
        # have used before in non-FL projects. For more advance FL implementation
        # you might want to tweak it but overall, from a client perspective the "local
        # training" can be seen as a form of "centralised training" given a pre-trained
        # model (i.e. the model received from the server)
        train(self.model, self.trainloader, self.optimizer, epochs, self.device)
        self.scheduler.step()

        # Flower clients need to return three arguments: the updated model, the number
        # of examples in the client (although this depends a bit on your choice of aggregation
        # strategy), and a dictionary of metrics (here you can add any additional data, but these
        # are ideally small data structures)
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)
        loss, f1, precision, recall = test(self.model, self.valloader, self.device)

        return float(loss), len(self.valloader), {"F1": f1, "Precision": precision, "Recall": recall}



def generate_client_fn(trainloaders, valloaders, num_classes, config):
    """Return a function that can be used by the VirtualClientEngine.

    to spawn a FlowerClient with client id `cid`.
    """

    def client_fn(cid: str):
        # This function will be called internally by the VirtualClientEngine
        # Each time the cid-th client is told to participate in the FL
        # simulation (whether it is for doing fit() or evaluate())

        # Returns a normal FLowerClient that will use the cid-th train/val
        # dataloaders as it's local data.
        return FlowerClient(
            trainloader=trainloaders[int(cid)],
            valloader=valloaders[int(cid)],
            num_classes=num_classes,
            config=config
        ).to_client()

    # return the function to spawn client
    return client_fn