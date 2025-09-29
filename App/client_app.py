# App/client_app.py

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

# Correct the import path to match the project structure
from App.task import (
    Autoencoder,
    evaluate,
    get_weights,
    get_data,
    set_weights,
    train,
    get_input_dim,
)


class FlowerClient(NumPyClient):
    """A Flower client for the anomaly detection task."""
    def __init__(self, net, trainloader, testloader):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader

    def fit(self, parameters, config):
        """Train the model on the client's local data."""
        set_weights(self.net, parameters)
        train(self.net, self.trainloader)
        return get_weights(self.net), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the global test set."""
        set_weights(self.net, parameters)
        loss, num_examples = evaluate(self.net, self.testloader)
        return float(loss), num_examples, {"loss": float(loss)}


def client_fn(context: Context) -> ClientApp:
    """A function to create a Flower client representing a single data partition."""
    # Get the partition ID from the context
    partition_id = context.node_config.get("partition-id", 0)

    # Load the client's data partition
    train_loader, test_loader = get_data(partition_id=partition_id)

    # Get the model's input dimension dynamically
    input_dim = get_input_dim()
    net = Autoencoder(input_dim=input_dim)

    # Create and return a Flower client
    return FlowerClient(net, train_loader, test_loader).to_client()


# Define the Flower ClientApp
app = ClientApp(client_fn=client_fn)