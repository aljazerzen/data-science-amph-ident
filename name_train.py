import torch
import os
import numpy as np
import math

import dataset
import model
import name_test

def append_progress(file, values):
    f = open(file, 'a')
    f.write(",".join(map(str, values)) + '\n')
    f.close()

def train(epochs = 100):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load
    net = model.AmphiNameNet()
    net.load()
    net.to(device)

    # Define a Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    (ds_train, loader_train), (ds_test, loader_test) = dataset.load_by_name(16, augmentation=True)

    # train
    for epoch in range(epochs):  # loop over the dataset multiple times

        # save & eval
        if epoch % 5 == 0:
            net.save()

            rank_freq_train, _ = name_test.evaluate(net, device, loader_train)
            rank_freq_test, _ = name_test.evaluate(net, device, loader_test)

            print(rank_freq_train, rank_freq_test)

            append_progress('rank_freqs.csv', [epoch, 'train'] + rank_freq_train)
            append_progress('rank_freqs.csv', [epoch, 'test'] + rank_freq_test)

        losses = []
        for i, (inputs, labels) in enumerate(loader_train, 0):
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            # print(inputs.size(), outputs.size(), labels.size())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # statistics
            losses.append(loss.item())
            if math.isnan(loss.item()):
                print('loss = nan, exiting')
                exit()
        
        loss_mean = np.mean(np.array(losses))
        loss_std = np.std(np.array(losses))
        print(f'[ep {epoch}] loss = {round(loss_mean, 3)} (std={round(loss_std, 2)})')
        append_progress('loss.csv', [epoch, loss_mean, loss_std])

    net.save()
    print('Finished Training')

if __name__ == '__main__':
    train(10000)
