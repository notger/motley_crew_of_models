import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from vae import VariationalAutoEncoder


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Generate the data to work on and split it into training and testing sets:
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=5, n_redundant=10, n_clusters_per_class=2, random_state=19770521,
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
pass