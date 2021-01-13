import torch
import numpy as np

import dataset
import model
import viz

def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train, test, ident = dataset.load_by_name(batch_size=10)

    net = model.AmphiNameNet()
    net.load(device)
    net.disable_dropout()

    print('train dataset results:')
    rank_freq, ranks_train = evaluate(net, device, train[1])
    print_summary(rank_freq)

    print('test dataset results:')
    rank_freq, ranks_test = evaluate(net, device, test[1])
    print_summary(rank_freq)

    save_summary(ranks_train, ranks_test, len(train[0].classes))

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
    ], ranks

def print_summary(rank_freq):
    print(f'rank1 : {round(100 * rank_freq[1] / rank_freq[0], 2)}%')
    print(f'rank2 : {round(100 * rank_freq[2] / rank_freq[0], 2)}%')
    print(f'rank5 : {round(100 * rank_freq[3] / rank_freq[0], 2)}%')
    print(f'rank10: {round(100 * rank_freq[4] / rank_freq[0], 2)}%')

def save_summary(ranks_train, ranks_test, class_count):
    with open('ranks.csv', 'w') as f:
        f.write('rank,count,frequency,subset')
        save_ranks(f, ranks_train, class_count, 'train')
        save_ranks(f, ranks_test, class_count, 'test')

def save_ranks(f, ranks, class_count, subset):
    for rank in np.arange(1, class_count):
        count = np.sum(ranks <= rank)
        freq = count / len(ranks)
        f.write(f'{rank},{count},{freq},{subset}\n')

if __name__ == '__main__':
    test()
