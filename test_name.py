import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

import matplotlib.pyplot as plt
import numpy as np
import math

import dataset
import model
import viz

def test():

    batch_size = 10

    datasets, dataloaders, ds = dataset.load_by_name(batch_size)

    net = model.AmphiNameNet()
    net.load()

    print('train dataset results:')
    evaluate(net, ds, dataloaders['train'])

    print('test dataset results:')
    evaluate(net, ds, dataloaders['test'])

def evaluate(net, ds, dataloader):
    wrong = {
        'label': [],
        'predicted': [],
        'image': [],
    }

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in np.arange(0, labels.size(0))[(predicted != labels).numpy()]:
                wrong['label'].append(labels[i])
                wrong['predicted'].append(predicted[i])
                wrong['image'].append(images[i])

    print(f' Accuracy: {math.floor(1000 * correct / total) / 10}% ({correct}/{total})')

    w = len(wrong['label'])
    if w > 0:
        print(f' Wrong ({w}):')
        # print('       Label  Predicted')
        # print(' ----------------------')
        # for i in range(min(20, len(wrong['label']))):
        #     print('  %10s %10s' % (ds.classes[wrong['label'][i]], ds.classes[wrong['predicted'][i]]))
        
        # viz.imshow(torchvision.utils.make_grid(wrong['image']))


if __name__ == '__main__':
    test()
