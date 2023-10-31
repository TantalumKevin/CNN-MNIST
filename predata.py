import torchvision
import torchvision.datasets.mnist as mnist
import numpy as np, os, torch
from torch.utils.data import TensorDataset

def install_data():
    train_data=torchvision.datasets.MNIST(
        root='',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    test_data=torchvision.datasets.MNIST(
        root='',
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )

def norm_data():
    root = "./MNIST"
    raw = root + "/raw"
    norm = root + "/norm.npz"

    train_set = (
        mnist.read_image_file(os.path.join(raw, 'train-images-idx3-ubyte')),
        mnist.read_label_file(os.path.join(raw, 'train-labels-idx1-ubyte'))
            )
    test_set = (
        mnist.read_image_file(os.path.join(raw, 't10k-images-idx3-ubyte')),
        mnist.read_label_file(os.path.join(raw, 't10k-labels-idx1-ubyte'))
            )
    
    train_data = train_set[0].numpy()
    train_label = train_set[1].numpy()
    test_data = test_set[0].numpy()
    test_label = test_set[1].numpy()
    train_data_max = np.expand_dims(np.expand_dims(np.max(np.max(train_data,axis = 1),axis = 1),axis=1),axis=1)

    test_data_max = np.expand_dims(np.expand_dims(np.max(np.max(test_data,axis = 1),axis = 1),axis=1),axis=1)
    train_data_min = np.expand_dims(np.expand_dims(np.min(np.min(train_data,axis = 1),axis = 1),axis=1),axis=1)
    test_data_min = np.expand_dims(np.expand_dims(np.min(np.min(test_data,axis = 1),axis = 1),axis=1),axis=1)

    train_data = (train_data - train_data_min) / (train_data_max - train_data_min)
    test_data = (test_data - test_data_min) / (test_data_max - test_data_min)
    np.savez(norm,train_data=train_data,train_label=train_label,test_data=test_data,test_label=test_label)

def load_data(batch_size = 64):
    data = np.load("./MNIST/norm.npz")
    train_data = data["train_data"]
    train_label = data["train_label"]
    test_data = data["test_data"]
    test_label = data["test_label"]
    train_data = torch.tensor(train_data).float()
    train_label = torch.tensor(train_label).long()
    test_data = torch.tensor(test_data).float()
    test_label = torch.tensor(test_label).long()
    trainset = TensorDataset(train_data,train_label)
    testset = TensorDataset(test_data, test_label)
    
    if torch.cuda.is_available():
        pin_memory = True
    else:
        pin_memory = False
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=False,pin_memory=pin_memory)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False,pin_memory=pin_memory)
    return trainloader, testloader

def predata():
    install_data()
    if not os.path.exists("./MNIST/norm.npz"):
        norm_data()
    else:
        print("Data file has existed.")
        
    return load_data()
    
if __name__ == "__main__":
    predata()