import torch
import os

import dataset
import model

def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load
    net = model.AmphiOrderNet()
    if os.path.isfile(model.PATH):
        net.load_state_dict(torch.load(model.PATH))
    net.to(device)

    # Define a Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    datasets, dataloaders, dataset_all = dataset.load()

    # train
    print_per = 50
    for epoch in range(2):  # loop over the dataset multiple times

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
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / print_per))
                running_loss = 0.0

    print('Finished Training')

    # save
    torch.save(net.state_dict(), model.PATH)

if __name__ == '__main__':
    train()
