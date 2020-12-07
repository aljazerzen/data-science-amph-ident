import torch
import os

import dataset
import model

def train(epochs = 100, print_per = 100):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load
    net = model.AmphiNameNet()
    net.load()
    net.to(device)

    # Define a Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    datasets, dataloaders, dataset_all = dataset.load_by_name()

    # train
    print_per = min(print_per, len(datasets['train']))
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(dataloaders['train'], 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % print_per == print_per - 1:
                print('[%3d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / print_per))
                running_loss = 0.0

        # save
        if epoch % 10 == 0:
            net.save()

    net.save()
    print('Finished Training')

if __name__ == '__main__':
    train()
