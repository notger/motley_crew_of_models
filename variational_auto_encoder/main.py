import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from vae import VariationalAutoEncoder


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameter section:
N_SAMPLES = 10_000
N_FEATURES = 20
N_INFORMATIVE = 5
N_REDUNDANT = 10
N_CLUSTERS_PER_CLASS = 2
N_EMBEDDINGS = 5
N_EPOCHS = 200

PLOT_AFTER_EACH_RUN = False
CONVERT_TO_DF = True


def generate_synthetic_data(n_samples, n_features, n_informative, n_redundant, n_clusters_per_class):
    # Generate the data to work on and split it into training and testing sets:
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_redundant=n_redundant,
        n_clusters_per_class=n_clusters_per_class, random_state=19770521,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19770521)
    mmscaler = MinMaxScaler()
    mmscaler.fit(X_train)
    X_train = mmscaler.transform(X_train)
    X_test = mmscaler.transform(X_test)


    # Transform the data into PyTorch tensors:
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_set = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_set = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    return train_loader, test_loader


# Define the loss function:
def loss(x, x_hat):
    """Simple quadratic loss function on the reconstruction quality."""
    return torch.mean((x - x_hat) ** 2)


# Define the training loop:
def train(vae, epoch):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(DEVICE)
        optimiser.zero_grad()
        reconstruction_e, _ = vae(data)
        L = loss(reconstruction_e, data)
        L.backward()
        train_loss += L.item()
        optimiser.step()

    print(f"\nEpoch {epoch}: Average loss: {train_loss / len(train_loader.dataset)}")

    return train_loss / len(train_loader.dataset)


# Define the test function:
def test(vae, epoch):
    vae.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):
            data = data.to(DEVICE)
            reconstruction_e, _ = vae(data)
            test_loss += loss(reconstruction_e, data)

    print(f"Test loss epoch {epoch}: {test_loss / len(test_loader.dataset)}")

    return test_loss / len(test_loader.dataset)


if __name__ == '__main__':
    # Generate the data:
    train_loader, test_loader = generate_synthetic_data(N_SAMPLES, N_FEATURES, N_INFORMATIVE, N_REDUNDANT, N_CLUSTERS_PER_CLASS)

    df = None

    # Run the "study":
    for n_embeddings in [1, N_INFORMATIVE // 2, N_INFORMATIVE, N_INFORMATIVE * 2, N_FEATURES]:
        # Set up the model:
        vae = VariationalAutoEncoder(num_features=N_FEATURES, num_embeddings=n_embeddings).to(DEVICE)
        optimiser = torch.optim.Adam(vae.parameters(), lr=0.001)

        # Train the model:
        train_losses = {}
        test_losses = {}
        for k in range(N_EPOCHS):
            train_losses[k] = train(vae, k)
            test_losses[k] = test(vae, k)

        if PLOT_AFTER_EACH_RUN:
            plt.plot(train_losses.keys(), train_losses.values(), test_losses.keys(), test_losses.values())
            plt.legend(['train', 'test'])
            plt.show()

        if CONVERT_TO_DF:
            dat = {
                "k": list(train_losses.keys()),
                "n_embeddings": [n_embeddings] * N_EPOCHS,
                "train_losses": list(train_losses.values()),
                "test_losses": [float(x) for x in list(test_losses.values())],  # For some reason, we need to enforce floats here. :shrug:
            }
            if df is None:
                df = pd.DataFrame(dat)
            else:
                df = pd.concat([df, pd.DataFrame(dat)], ignore_index=True)

    if df is not None:
        filename = f"vae_study_n_informative_{N_INFORMATIVE}.csv"
        df.to_csv(filename)
        print(f"Stored the results for {N_EPOCHS} in '{filename}' and embedding sizes {df.n_embeddings.unique()}.")
