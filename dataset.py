import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
import math

def load(batch_size = 1): 

    data_transforms = transforms.Compose([
        transforms.Resize(400),
        transforms.CenterCrop([400, 200]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


    dataset_all = torchvision.datasets.ImageFolder('./Amphibious dataset/by_order', data_transforms)
    train = math.floor(len(dataset_all) * 0.7)

    datasets = random_split(
        dataset_all, 
        [train, len(dataset_all) - train], 
        generator=torch.Generator().manual_seed(42)
    )
    datasets = { 'train': datasets[0], 'test': datasets[1] }

    dataloaders = {
        x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,
                                        shuffle=True, num_workers=4)
                for x in ['train', 'test']
    }

    return datasets, dataloaders, dataset_all