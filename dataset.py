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

def load_by_name_belly(batch_size = 1, augementation = False): 
    torch.manual_seed(17)

    data_transforms = []
    if augementation:
        data_transforms.extend([
            transforms.ColorJitter((0.4, 1.7), 0.2, 0.4, 0.04),
            
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.95, 1.05), shear=5),
        ])

    data_transforms.extend([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    data_transforms = transforms.Compose(data_transforms)

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
  datasets, dataloaders, dataset = load_by_name_belly(16)
  
  images = []
  for i in range(8):
    loader_iter = iter(torch.utils.data.DataLoader(dataset, batch_size=16))
    # loader_iter = iter(dataloaders['test'])
    [imgs, labels] = loader_iter.next()
    images.extend(imgs)

  # print labels
#   print(' '.join('%5s' % dataset.classes[labels[j]] for j in range(10)))
  # show images
  viz.imshow(torchvision.utils.make_grid(images))
  
  