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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train, test = dataset.load_by_name(batch_size=10)

    net = model.AmphiNameNet()
    net.load()
    net.to(device)

    print('train dataset results:')
    print_summary(evaluate(net, device, train[1]))

    print('test dataset results:')
    print_summary(evaluate(net, device, test[1]))

def evaluate(net, device, dataloader):
    
    ranks = []
    correct = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)

            outputs = net(inputs).cpu()

            for i in range(labels.size(0)):
                label = labels[i]
                output = outputs[i, label].numpy()
                rank = np.sum((outputs[i,:].numpy() >= output))
                
                if rank == 0:
                    correct.append(label)
                ranks.append(rank)
    
    ranks = np.array(ranks)
    return [
        len(ranks),
        np.sum(ranks <= 1),
        np.sum(ranks <= 2),
        np.sum(ranks <= 5),
        np.sum(ranks <= 10),
    ]

def print_summary(rank_freqs):
    print(f'rank1 : {round(100 * rank_freqs[1] / rank_freqs[0], 2)}%')
    print(f'rank2 : {round(100 * rank_freqs[2] / rank_freqs[0], 2)}%')
    print(f'rank5 : {round(100 * rank_freqs[3] / rank_freqs[0], 2)}%')
    print(f'rank10: {round(100 * rank_freqs[4] / rank_freqs[0], 2)}%')

if __name__ == '__main__':
    test()
