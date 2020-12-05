import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

import matplotlib.pyplot as plt
import numpy as np

import dataset
import model
import viz

# print("torch.cuda.is_available: " + str(torch.cuda.is_available()))

def test():

    batch_size = 10

    datasets, dataloaders, ds = dataset.load(batch_size)

    # print images

    net = model.AmphiOrderNet()
    net.load_state_dict(torch.load(model.PATH))

    wrong = {
        'label': [],
        'predicted': [],
        'image': [],
    }

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloaders['test']:
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in np.arange(0, labels.size(0))[(predicted != labels).numpy()]:
                wrong['label'].append(labels[i])
                wrong['predicted'].append(predicted[i])
                wrong['image'].append(images[i])

    print(f'Accuracy: {100 * correct / total}% ({correct}/{total})')

    w = len(wrong['label'])
    if w > 0:
        print(f'Wrong ({w}):')
        print(' labels:    ', ' '.join('%5s' % ds.classes[wrong['label'][j]] for j in range(w)))
        print(' predicted: ', ' '.join('%5s' % ds.classes[wrong['predicted'][j]] for j in range(w)))
        viz.imshow(torchvision.utils.make_grid(wrong['image']))


if __name__ == '__main__':
    test()
