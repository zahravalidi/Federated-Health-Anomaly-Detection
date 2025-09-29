# App/server_app.py

import flwr as fl
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedProx
import pickle

# Correct the import path to match the project structure
from App.task import Autoencoder, get_weights, get_input_dim

# Define the path where the final model will be saved
SAVED_MODEL_PATH = "Generated Artifacts/final_model_fedprox.pkl"


class SaveModelFedProxStrategy(FedProx):
    """A FedProx strategy that saves the final global model."""
    def __init__(self, *args, **kwargs):
        self.num_rounds = kwargs.pop("num_rounds")
        super().__init__(*args, **kwargs)

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate fit results and save the final model."""
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None and server_round == self.num_rounds:
            print(f"--- Saving final global model (FedProx) at round {server_round} ---")
            weights = parameters_to_ndarrays(aggregated_parameters)
            with open(SAVED_MODEL_PATH, "wb") as f:
                pickle.dump(weights, f)

        return aggregated_parameters, aggregated_metrics


def weighted_average_loss(metrics):
    """A function to compute the weighted average loss from client evaluation metrics."""
    total_examples = sum(num_examples for num_examples, _ in metrics)
    weighted_loss = sum(num_examples * m["loss"] for num_examples, m in metrics)
    avg_loss = weighted_loss / total_examples if total_examples > 0 else 0
    print(f"Round weighted average loss (FedProx): {avg_loss}")
    return {"loss": avg_loss}


def server_fn(context: Context) -> ServerAppComponents:
    """A function to define and return the components of the server app."""
    # Get the model's input dimension dynamically from the preprocessed data
    input_dim = get_input_dim()
    print(f"--- Server: Building model with input_dim={input_dim} ---")

    # Initialize the global model
    net = Autoencoder(input_dim=input_dim)
    params = ndarrays_to_parameters(get_weights(net))
    num_rounds = context.run_config.get("num_server_rounds", 20)

    # Define the federated learning strategy
    strategy = SaveModelFedProxStrategy(
        initial_parameters=params,
        evaluate_metrics_aggregation_fn=weighted_average_loss,
        num_rounds=num_rounds,
        proximal_mu=0.1  # Example value, can be tuned
    )

    # Configure the server
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(config=config, strategy=strategy)

# Define the Flower ServerApp
app = ServerApp(server_fn=server_fn)