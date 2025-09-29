# App/task.py

from collections import OrderedDict
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

# Define the path to the preprocessed data artifact
# This path is relative to the project's root directory
DATA_ARTIFACT_PATH = "Generated Artifacts/preprocessed_unsupervised.pkl"

# Global caches for data partitions
CLIENT_PARTITIONS = None
TEST_DATA = None


def get_input_dim():
    """Reads the input dimension from the preprocessed data file."""
    try:
        with open(DATA_ARTIFACT_PATH, "rb") as f:
            processed_data = pickle.load(f)
        return processed_data["input_dim"]
    except (FileNotFoundError, KeyError):
        raise RuntimeError(
            f"Could not load input_dim from '{DATA_ARTIFACT_PATH}'. "
            "Please run the main notebook first to generate this file."
        )


def load_and_partition_data():
    """Loads preprocessed data and creates client partitions."""
    global CLIENT_PARTITIONS, TEST_DATA

    with open(DATA_ARTIFACT_PATH, "rb") as f:
        processed_data = pickle.load(f)

    X_train = processed_data["X_train"]
    X_train_scaled = processed_data["X_train_scaled"]

    X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train.index)
    X_train_with_city = X_train_scaled_df.join(X_train['city'])

    unique_cities = X_train['city'].unique()
    client_dfs = []
    for city_id in unique_cities:
        client_df = X_train_with_city[X_train_with_city['city'] == city_id].drop('city', axis=1)
        client_dfs.append(client_df)

    CLIENT_PARTITIONS = [p.values for p in client_dfs if not p.empty]
    TEST_DATA = {
        "X_test_scaled": processed_data["X_test_scaled"],
        "X_test_original": processed_data["X_test"]
    }


def get_data(partition_id: int):
    """Get a client's data partition and the global test set."""
    if CLIENT_PARTITIONS is None:
        load_and_partition_data()

    client_data = CLIENT_PARTITIONS[partition_id]
    client_tensor = torch.tensor(client_data, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(client_tensor), batch_size=32, shuffle=True, drop_last=True)

    test_tensor = torch.tensor(TEST_DATA["X_test_scaled"], dtype=torch.float32)
    test_loader = DataLoader(TensorDataset(test_tensor), batch_size=32)

    return train_loader, test_loader


class Autoencoder(nn.Module):
    """A simple Autoencoder model for anomaly detection."""
    def __init__(self, input_dim: int):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train(model, train_loader, num_epochs=5):
    """Train the model on the provided data loader."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for _ in range(num_epochs):
        for data in train_loader:
            inputs = data[0]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()


def evaluate(model, test_loader):
    """Evaluate the model on the test set."""
    criterion = nn.MSELoss()
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs = data[0]
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0
    return avg_loss, len(test_loader.dataset)


def set_weights(net, parameters):
    """Set the weights of a model from a list of NumPy arrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_weights(net):
    """Get the weights of a model as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]