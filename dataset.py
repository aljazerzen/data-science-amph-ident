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

    dataset = torchvision.datasets.ImageFolder('./Amphibian dataset/by_order', data_transforms)
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

    a = f'amphibian dataset (size={len(dataset)}, classes={len(dataset.classes)}, train={len(subsets["train"])}, test={len(subsets["test"])})'
    print(a)

    return subsets, dataloaders, dataset


def load_by_name_subset(batch_size, augmentation, is_valid_file):
    # prepare tranformations
    perturbations = [
        transforms.ColorJitter((0.8, 1.2), 0.2, 0.4, 0.04),
        transforms.RandomAffine(degrees=2, translate=(0.05, 0.05), scale=(0.998, 1.02), shear=1),
    ] if augmentation else []

    data_transforms = transforms.Compose(perturbations + [
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = torchvision.datasets.ImageFolder('./Amphibian dataset/by_name/', data_transforms, is_valid_file=is_valid_file)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    return dataset, loader

def load_by_name(batch_size = 1, augmentation = False, include_test = True): 
    torch.manual_seed(17)

    test_images = open('test-images.txt').readlines()
    test_images = list(map(lambda line: line.replace('\n', ''), test_images))
    is_test = lambda path: path.split('by_name/')[1] in test_images
    is_train = lambda path: not is_test(path)

    train = load_by_name_subset(batch_size, augmentation, is_train)
    test = load_by_name_subset(batch_size, False, is_test) if include_test else None

    print(f'amphibian dataset ('+', '.join(
        [f'classes={len(train[0].classes)}'] +
        [f'train_cases={len(train[0])}'] +
        ([
            f'test_cases={len(test[0])} ({round(100 * len(test[0]) /(len(train[0]) + len(test[0])), 1)}%)', 
            f'total_cases={len(train[0]) + len(test[0])}', 
        ] if test != None else []) +
        ['augmented'] if augmentation else []
    ) + ')')

    return train, test


if __name__ == "__main__":
  (train_ds, train_loader), test = load_by_name(1, True)
  
  images = []
  for i in range(8):
    loader_iter = iter(torch.utils.data.DataLoader(train_ds, batch_size=8))
    # loader_iter = iter(dataloaders['test'])
    [imgs, labels] = loader_iter.next()
    images.extend(imgs)

  # print labels
  # print(' '.join('%5s' % dataset.classes[labels[j]] for j in range(10)))

  # show images
  viz.imshow(torchvision.utils.make_grid(images))
