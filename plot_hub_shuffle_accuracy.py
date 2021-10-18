import torch
from torch.nn import Sequential, Linear, ReLU
import torchvision

from hub.constants import KB, MB


TORCHVISION_MNIST = "./torchvision_mnist"


EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.01
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_model():
    model = Sequential(
        Linear(784, 800),      # hidden layer 1 weights
        ReLU(),
        Linear(800, 800),      # hidden layer 2 weights
        ReLU(),
        Linear(800, 10),       # output layer weights
    )

    return model.to(DEVICE)


def get_optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)


def train(model, optimizer, loader, test_loader, name):
    """Train model and return best found test accuracy."""

    best_accuracy = 0
    for epoch in range(EPOCHS):
        model.train()

        print(f"training epoch {epoch} for {name}")

        for X, T in loader:
            X = X.flatten(start_dim=1).to(DEVICE)
            T = T.to(DEVICE)

            optimizer.zero_grad()

            Y = model(X)

            loss = torch.nn.functional.cross_entropy(Y, T)
            loss.backward()

            optimizer.step()

        test_accuracy = test(model, test_loader)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
    
    return best_accuracy


def test(model, loader):
    model.eval()

    correct = 0
    total = 0
    for X, T in loader:
        X = X.flatten(start_dim=1).to(DEVICE)
        T = T.to(DEVICE)

        Y = model(X)

        pred = torch.argmax(Y, dim=1)
        correct += (pred == T).sum().item()
        total += X.size(0)

    return correct / total


def train_mnist_torchvision(train_shuffle: bool=True) -> float:
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.MNIST(TORCHVISION_MNIST, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(TORCHVISION_MNIST, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=train_shuffle)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = get_model()
    optimizer = get_optimizer(model)
    return train(model, optimizer, train_loader, test_loader, f"torchvision, shuffle={train_shuffle}") 


def train_mnist_hub(shuffle_cache_size: int):
    raise NotImplementedError


if __name__ == "__main__":
    shuffle_cache_sizes = [1 * KB, 8 * MB, 16 * MB]
    
    tv_shuffle_accuracy = train_mnist_torchvision(train_shuffle=True)
    print(f"torchvision, shuffle=True: {tv_shuffle_accuracy}")
    tv_no_shuffle_accuracy = train_mnist_torchvision(train_shuffle=False)
    print(f"torchvision, shuffle=False: {tv_no_shuffle_accuracy}")

    for shuffle_cache_size in shuffle_cache_sizes:
        pass

    # TODO: plot accuracy vs cache size
    # TODO: plot shuffle-off accuracy
    # TODO: plot (non-hub) pytorch shuffle accuracy