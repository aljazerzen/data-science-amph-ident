import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
import math
import viz

def load_by_order(batch_size = 1): 

    data_transforms = transforms.Compose([
        transforms.Resize(400),
        transforms.CenterCrop([400, 200]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = torchvision.datasets.ImageFolder('./Amphibious dataset/by_order', data_transforms)
    train = math.floor(len(dataset) * 0.7)

    subsets = random_split(
        dataset, 
        [train, len(dataset) - train], 
        generator=torch.Generator().manual_seed(42)
    )
    subsets = { 'train': subsets[0], 'test': subsets[1] }

    dataloaders = {
        x: torch.utils.data.DataLoader(subsets[x], batch_size=batch_size,
                                        shuffle=True, num_workers=4)
                for x in ['train', 'test']
    }

    a = f'amphibious dataset (size={len(dataset)}, classes={len(dataset.classes)}, train={len(subsets["train"])}, test={len(subsets["test"])})'
    print(a)

    return subsets, dataloaders, dataset

def load_by_name(batch_size = 1): 

    data_transforms = transforms.Compose([
        transforms.Resize(400),
        transforms.CenterCrop([400, 200]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = torchvision.datasets.ImageFolder('./Amphibious dataset/by_name', data_transforms)
    train = math.floor(len(dataset) * 0.7)

    subsets = random_split(
        dataset, 
        [train, len(dataset) - train], 
        generator=torch.Generator().manual_seed(41)
    )
    subsets = { 'train': subsets[0], 'test': subsets[1] }

    dataloaders = {
        x: torch.utils.data.DataLoader(
            subsets[x], batch_size=batch_size, shuffle=True, num_workers=4
        ) for x in ['train', 'test']
    }

    a = f'amphibious dataset (size={len(dataset)}, classes={len(dataset.classes)}, train={len(subsets["train"])}, test={len(subsets["test"])})'
    print(a)

    return subsets, dataloaders, dataset

def load_by_name_belly(batch_size = 1): 

    data_transforms = transforms.Compose([
        transforms.Resize(400),
        transforms.CenterCrop([400, 200]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = torchvision.datasets.ImageFolder('./Amphibious dataset/by_name_belly', data_transforms)
    train = math.floor(len(dataset) * 0.7)

    subsets = random_split(
        dataset, 
        [train, len(dataset) - train], 
        generator=torch.Generator().manual_seed(41)
    )
    subsets = { 'train': subsets[0], 'test': subsets[1] }

    dataloaders = {
        x: torch.utils.data.DataLoader(
            subsets[x], batch_size=batch_size, shuffle=True, num_workers=4
        ) for x in ['train', 'test']
    }

    a = f'amphibious dataset (size={len(dataset)}, classes={len(dataset.classes)}, train={len(subsets["train"])}, test={len(subsets["test"])})'
    print(a)

    return subsets, dataloaders, dataset


if __name__ == "__main__":
  datasets, dataloaders, dataset = load_by_name_belly(10)
  
  dataiter = iter(dataloaders['train'])
  images, labels = dataiter.next()

  # show images
  viz.imshow(torchvision.utils.make_grid(images))
  # print labels
  print(' '.join('%5s' % dataset.classes[labels[j]] for j in range(10)))