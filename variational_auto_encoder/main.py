import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from vae import VariationalAutoEncoder


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameter section:
N_SAMPLES = 1000
N_FEATURES = 20
N_INFORMATIVE = 5
N_REDUNDANT = 10
N_CLUSTERS_PER_CLASS = 2
N_EMBEDDINGS = 5


# Generate the data to work on and split it into training and testing sets:
X, y = make_classification(
    n_samples=N_SAMPLES, n_features=N_FEATURES, n_informative=N_INFORMATIVE, n_redundant=N_REDUNDANT,
    n_clusters_per_class=N_REDUNDANT, random_state=19770521,
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


# Set up the model:
vae = VariationalAutoEncoder(num_features=N_FEATURES, num_embeddings=N_EMBEDDINGS).to(DEVICE)
optimiser = torch.optim.Adam(vae.parameters(), lr=0.001)


# Define the loss function:
def loss(x, x_hat):
    """Simple quadratic loss function on the reconstruction quality."""
    return torch.mean((x - x_hat) ** 2)


# Define the training loop:
def train(epoch):
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

    print(f"Epoch {epoch}: Average loss: {train_loss / len(train_loader.dataset)}")

# Define the test function:
def test(epoch):
    vae.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):
            data = data.to(DEVICE)
            reconstruction_e, _ = vae(data)
            test_loss += loss(reconstruction_e, data)

    print(f"Test loss epoch {epoch}: {test_loss / len(test_loader.dataset)}")


if __name__ == '__main__':
    pass
