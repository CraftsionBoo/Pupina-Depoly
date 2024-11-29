import os 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get(batch_size, data_root = "./", train=True, val=True, **kwards):
    data_root = os.path.expanduser(os.path.join(data_root, "mnist"))
    num_workers = kwards.setdefault("num_workers", 1)
    print("Building MNIST data loader with {} workers".format(num_workers))
    
    ds = []
    if train:
        train_loader = DataLoader(
            datasets.MNIST(root=data_root, train=True, download=True, 
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])), 
                           batch_size=batch_size, shuffle=True, **kwards)
        ds.append(train_loader)
    if val:
        test_loader = DataLoader(
            datasets.MNIST(root=data_root, train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
                           batch_size=batch_size, shuffle=True, **kwards)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds 
    return ds