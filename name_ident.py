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

approach = 'retrained-fc1-relu-msr'

def identify():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train, test, ident = dataset.load_by_name(batch_size=10, include_test=True, include_ident=True)

    net = model.AmphiNameNet()
    net.load(device)

    train_labels, train_outputs = evaluate(net, device, train[1])
    test_labels, test_outputs = evaluate(net, device, test[1])
    ident_labels, ident_outputs = evaluate(net, device, ident[1])

    cases = [(train_labels[i], train_outputs[i, :], 'train') for i in range(len(train_labels))] +\
        [(test_labels[i], test_outputs[i, :], 'test') for i in range(len(test_labels))] +\
        [(ident_labels[i], ident_outputs[i, :], 'ident') for i in range(len(ident_labels))]
    
    matches, ranks = match_each_to_each(cases)

    append_csv(matches, 'matches.csv')
    save_summary(ranks, len(cases))

def evaluate(net, device, dataloader):
    all_labels = np.array([])
    all_outputs = np.empty((0, 256))

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)

            outputs = net(inputs).cpu().numpy()
            
            all_labels = np.concatenate((all_labels, labels.numpy()))
            all_outputs = np.concatenate((all_outputs, outputs))
    return all_labels, all_outputs


def match_each_to_each(cases):
    matches = []
    ranks = []

    for i in range(len(cases)):
        a_label, a_fv, a_subset = cases[i]

        distances_inter = []
        distances_intra = []
        for j in range(len(cases)):
            if i == j: continue
            b_label, b_fv, b_subset = cases[j]

            # cosine distance
            # distance = np.dot(a_fv, b_fv) / math.sqrt(np.sum(a_fv ** 2)) / math.sqrt(np.sum(b_fv ** 2))

            # msr distance
            distance = np.sum((a_fv - b_fv) ** 2)
            
            if a_label == b_label:
                distances_intra.append(distance)
            else:
                distances_inter.append(distance)

            matches.append([i, a_label, a_subset, j, b_label, b_subset, distance])
        
        if len(distances_intra) > 0:
            min_intra = np.min(np.array(distances_intra))
            rank = np.sum(distances_inter < min_intra) + 1
            ranks.append([rank, a_subset])

    return matches, ranks

def append_csv(matches, filename):
    with open(filename, 'a') as f:
        for match in matches:
            f.write(','.join(map(str, match + [approach])) + '\n')

def save_summary(rank_matches, class_count):
    with open('ranks-ident.csv', 'a') as f:
        for subset in ['train', 'test', 'ident']:
            ranks = [rm[0] for rm in rank_matches if rm[1] == subset]
            save_ranks(f, ranks, class_count, subset)

def save_ranks(f, ranks, class_count, subset):
    for rank in np.arange(1, class_count):
        count = np.sum(ranks <= rank)
        freq = count / len(ranks)
        f.write(f'{rank},{count},{freq},{subset},{approach}\n')

if __name__ == '__main__':
    identify()
