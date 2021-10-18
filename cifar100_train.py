import ray

import hub

import torch
import torchvision
import torchvision.transforms as tt

from tqdm import tqdm

# stole hyperparameters from:
# https://blog.jovian.ai/classifying-cifar-100-with-resnet-5860a9c2c13f

# define torchvision dataset
TORCHVISION_DATA_DIRECTORY = "./torchvision_dataset"
TORCHVISION_DATA_CLASS = torchvision.datasets.CIFAR100

# define hub dataset
HUB_TRAIN_URI = "hub://activeloop/cifar100-train"
HUB_TEST_URI = "hub://activeloop/cifar100-test"

EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
WORKERS = 2

def get_model():
    model = torchvision.models.resnet50(pretrained=True)

    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 100)
    
    return model.to(DEVICE)

def get_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train(model, optimizer, loader, test_loader, name):
    """Train model and return best found test accuracy."""

    best_accuracy = 0
    for epoch in range(EPOCHS):
        model.train()

        for X, T in tqdm(loader, total=len(loader), desc=f"training epoch {epoch} for {name}"):
            X = X.to(DEVICE)
            T = T.to(DEVICE)

            optimizer.zero_grad()

            Y = model(X)

            loss = torch.nn.functional.cross_entropy(Y, T)
            loss.backward()

            optimizer.step()

        test_accuracy = test(model, test_loader)
        print(f"epoch {epoch} test accuracy: {test_accuracy}")
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
    
    return best_accuracy


@torch.no_grad()
def test(model, loader):
    model.eval()

    correct = 0
    total = 0
    for X, T in tqdm(loader, total=len(loader), desc=f"testing"):
        X = X.to(DEVICE)
        T = T.to(DEVICE)

        Y = model(X)

        pred = torch.argmax(Y, dim=1)
        correct += (pred == T).sum().item()
        total += X.size(0)

    return correct / total


stats = ((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
train_tfs = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'), 
                         tt.RandomHorizontalFlip(), 
                         tt.ToTensor(), 
                         tt.Normalize(*stats,inplace=True)])
test_tfs = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])


@ray.remote
def train_cifar_torchvision(train_shuffle: bool=True) -> float:
    train_dataset = TORCHVISION_DATA_CLASS(TORCHVISION_DATA_DIRECTORY, train=True, download=True, transform=train_tfs)
    test_dataset = TORCHVISION_DATA_CLASS(TORCHVISION_DATA_DIRECTORY, train=False, download=True, transform=test_tfs)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=train_shuffle, num_workers=WORKERS)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)

    model = get_model()
    optimizer = get_optimizer(model)
    return train(model, optimizer, train_loader, test_loader, f"torchvision, shuffle={train_shuffle}") 


@ray.remote
def train_cifar_hub(train_shuffle: bool=True) -> float:
    train_dataset = hub.load(HUB_TRAIN_URI)
    test_dataset = hub.load(HUB_TEST_URI)

    # train_dataset = torchvision.datasets.CIFAR100(TORCHVISION_CIFAR, train=True, download=True, transform=train_tfs)
    # test_dataset = torchvision.datasets.CIFAR100(TORCHVISION_CIFAR, train=False, download=True, transform=test_tfs)

    def hub_train_tfs(sample):
        image, label = sample["images"], sample["fine_labels"]
        image = train_tfs(image)
        label = torch.tensor(label)
        return image, label

    def hub_test_tfs(sample):
        image, label = sample["images"], sample["fine_labels"]
        image = test_tfs(image)
        label = torch.tensor(label)
        return image, label

    train_loader = train_dataset.pytorch(batch_size=BATCH_SIZE, shuffle=train_shuffle, num_workers=WORKERS, transform=hub_train_tfs)
    test_loader = test_dataset.pytorch(batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, transform=hub_test_tfs)

    model = get_model()
    optimizer = get_optimizer(model)
    return train(model, optimizer, train_loader, test_loader, f"torchvision, shuffle={train_shuffle}") 


if __name__ == "__main__":
    # best case scenario
    # tv_shuffle = train_cifar_torchvision.remote(train_shuffle=True)
    
    # worst case scenario
    # tv_no_shuffle = train_cifar_torchvision.remote(train_shuffle=False)

    # print(ray.get([tv_shuffle, tv_no_shuffle]))

    hub_no_shuffle = train_cifar_hub.remote(train_shuffle=False)
    print(ray.get(hub_no_shuffle))