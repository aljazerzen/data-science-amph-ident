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
        transforms.ColorJitter((0.4, 1.7), 0.2, 0.4, 0.04),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        # transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.95, 1.05), shear=1),
    ] if augmentation else []

    data_transforms = transforms.Compose([
        transforms.Resize([256, 256]),
    ] + perturbations + [
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = torchvision.datasets.ImageFolder('./Amphibian dataset/by_name/', data_transforms, is_valid_file=is_valid_file)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    return dataset, loader

def load_text_file(filename):
    lines = open(filename).readlines()
    return list(map(lambda line: line.replace('\n', ''), lines))

def load_by_name(batch_size = 1, augmentation = False, include_test = True, include_ident = False):
    torch.manual_seed(17)

    test_images = load_text_file('images-test.txt')
    ident_images = load_text_file('images-ident.txt')
    is_test = lambda path: path.split('by_name/')[1] in test_images
    is_ident = lambda path: path.split('by_name/')[1] in ident_images
    is_train = lambda path: not is_test(path) and not is_ident(path)

    train = load_by_name_subset(batch_size, augmentation, is_train)
    test = load_by_name_subset(batch_size, False, is_test) if include_test else None
    ident = load_by_name_subset(batch_size, False, is_ident) if include_ident else None

    total_cases = len(train[0]) +\
        (len(test[0]) if include_test else 0) +\
        (len(ident[0]) if include_ident else 0)

    print(f'amphibian dataset ('+', '.join(
        [f'classes={len(train[0].classes)}'] +
        [f'train_cases={len(train[0])} ({round(100 * len(train[0]) / total_cases, 1)}%)'] +
        ([
            f'test_cases={len(test[0])} ({round(100 * len(test[0]) / total_cases, 1)}%)', 
        ] if test != None else []) +
        ([
            f'ident_cases={len(ident[0])} ({round(100 * len(ident[0]) / total_cases, 1)}%)', 
        ] if ident != None else []) +
        [f'total_cases={total_cases}'] +
        (['augmented'] if augmentation else [])
    ) + ')')

    return train, test, ident


if __name__ == "__main__":
  (train_ds, train_loader), test, ident = load_by_name(1, True, True, True)
  
  images = []
  for i in range(4):
    loader_iter = iter(torch.utils.data.DataLoader(train_ds, batch_size=8))
    # loader_iter = iter(dataloaders['test'])
    [imgs, labels] = loader_iter.next()
    images = images + list(imgs[0:3]) + list(imgs[7:8])

  # print labels
  # print(' '.join('%5s' % dataset.classes[labels[j]] for j in range(10)))

  # show images
  viz.imshow(torchvision.utils.make_grid(images, nrow = 4))
