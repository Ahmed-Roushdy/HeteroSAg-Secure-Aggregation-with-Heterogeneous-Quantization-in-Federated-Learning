import logging

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
import os
import h5py
import copy
from sklearn.utils import shuffle
from math import ceil
import time
import random
import train_utils
from torch.utils.data import Subset, DataLoader
import argparse
from train_utils import get_data_set
from Quantizer import *







def non_iid_partition_with_dirichlet_distribution(label_list,
                                                  client_num,alpha,
                                                  classes = 10,
                                                  task='classification'):
    """
        Obtain sample index list for each client from the Dirichlet distribution.
        This LDA method is first proposed by :
        Measuring the Effects of Non-Identical Data Distribution for
        Federated Visual Classification (https://arxiv.org/pdf/1909.06335.pdf).
        This can generate nonIIDness with unbalance sample number in each label.
        The Dirichlet distribution is a density over a K dimensional vector p whose K components are positive and sum to 1.
        Dirichlet can support the probabilities of a K-way categorical event.
        In FL, we can view K clients' sample number obeys the Dirichlet distribution.
        For more details of the Dirichlet distribution, please check https://en.wikipedia.org/wiki/Dirichlet_distribution
        Parameters
        ----------
            label_list : the label list from classification/segmentation dataset
            client_num : number of clients
            classes: the number of classification (e.g., 10 for CIFAR-10) OR a list of segmentation categories
            alpha: a concentration parameter controlling the identicalness among clients.
            task: CV specific task eg. classification, segmentation
        Returns
        -------
            samples : ndarray,
                The drawn samples, of shape ``(size, k)``.
    """
    net_dataidx_map = {}
    K = classes

    # For multiclass labels, the list is ragged and not a numpy array
    N = len(label_list) if task == 'segmentation' else label_list.shape[0]

    # guarantee the minimum number of sample in each client
    min_size = 0
    while min_size < 10:
        idx_batch = [[] for _ in range(client_num)]

        if task == 'segmentation':
            # Unlike classification tasks, here, one instance may have multiple categories/classes
            for c, cat in enumerate(classes):
                if c > 0:
                    idx_k = np.asarray([np.any(label_list[i] == cat) and not np.any(
                        np.in1d(label_list[i], classes[:c])) for i in
                                        range(len(label_list))])
                else:
                    idx_k = np.asarray(
                        [np.any(label_list[i] == cat) for i in range(len(label_list))])

                # Get the indices of images that have category = c
                idx_k = np.where(idx_k)[0]
                idx_batch, min_size = partition_class_samples_with_dirichlet_distribution(N, alpha, client_num,
                                                                                          idx_batch, idx_k)
        else:
            # for each classification in the dataset
            for k in range(K):
                # get a list of batch indexes which are belong to label k
                idx_k = np.where(label_list == k)[0]
                idx_batch, min_size = partition_class_samples_with_dirichlet_distribution(N, alpha, client_num,
                                                                                          idx_batch, idx_k)
    for i in range(client_num):
        np.random.shuffle(idx_batch[i])
        net_dataidx_map[i] = idx_batch[i]

    return net_dataidx_map


def partition_class_samples_with_dirichlet_distribution(N, alpha, client_num, idx_batch, idx_k):
    np.random.shuffle(idx_k)
    # using dirichlet distribution to determine the unbalanced proportion for each client (client_num in total)
    # e.g., when client_num = 4, proportions = [0.29543505 0.38414498 0.31998781 0.00043216], sum(proportions) = 1
    proportions = np.random.dirichlet(np.repeat(alpha, client_num))

    # get the index in idx_k according to the dirichlet distribution
    proportions = np.array([p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)])
    proportions = proportions / proportions.sum()
    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

    # generate the batch list for each client
    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
    min_size = min([len(idx_j) for idx_j in idx_batch])

    return idx_batch, min_size


def record_data_stats(y_train, net_dataidx_map, task='classification'):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(np.concatenate(y_train[dataidx]), return_counts=True) if task == 'segmentation' \
            else np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts






def get_train_clients_loaders(train_dataset,total_clients,training_batch_frac,label_list, alpha):
    """
    Shuffle the clients. Thus, create a list of benign clients (randomly) and consequentially a list of Byzantine clients.
    Use the permuted all clients list to create train data loader objects, so that data is allocated randomly among the benign and Byzantine clients.
    output: train_loaders and array containing status of clients (1 means good, i.e. benign)
    """
    train_indexes = non_iid_partition_with_dirichlet_distribution(label_list,
                                                  total_clients,alpha,
                                                  classes = 10,
                                                  task='classification')
    
  
    train_total_data_size = train_utils.get_dataset_num('cifar10')
    train_data_size_clients = np.zeros((total_clients))
    train_clients_loaders = {}
    for i_index in range(total_clients):
        worker_dataset_size = len(train_indexes[i_index])
        
        worker_train_batch_size = int(training_batch_frac*worker_dataset_size)
        train_clients_loaders[i_index] = DataLoader(Subset(train_dataset, train_indexes[i_index]), 
                                                batch_size=worker_train_batch_size, shuffle=True)
    return train_clients_loaders, train_indexes

def get_test_loader(test_dataset, batch_size=100, device='cpu'):
    # For faster implementation, created an in-memory ready to use list of batches of features and labels for test accuracy at each client 
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    test_images = []
    test_labels = []
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        test_images.append(inputs)
        test_labels.append(labels)
    test_dataset_size = len(test_loader.dataset)
    return test_images, test_labels, test_dataset_size






def main(args):

    alpha = args.alpha



    device = 'cuda:5' if torch.cuda.is_available() else 'cpu'







    Epochs_num = 10
    nClasses = 10
    nClients = 300
    trainTotalRounds = 300
    Batch_size = 0.2
    lrIni = 0.06
    lrFac = 1.0
    layer = 10 # the total number of weights  as tensors  and biases   ( Each layer   is counted as 2)



    seed = 1
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Segment_0_length = total number of parameter over G


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



    train_dataset, test_dataset = get_data_set('cifar10')
    data_size = 50000

    labels = []
    for _, label in train_dataset:
        labels.append(label)

    label_list = np.array(labels)


    train_clients_loaders, train_indexes = get_train_clients_loaders(train_dataset,nClients,Batch_size,label_list,alpha)
    test_images, test_labels,test_dataset_size = get_test_loader(test_dataset, batch_size=100, device=device)    


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    class CIFAR(nn.Module):
        def __init__(self, num_classes=10):
            super(CIFAR, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3),
                nn.Conv2d(16, 64, kernel_size=4, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=4),
            )
            self.classifier = nn.Sequential(
                nn.Linear(256, 384),
                nn.ReLU(inplace=True),
                nn.Linear(384, 192),
                nn.ReLU(inplace=True),
                nn.Linear(192, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), 256)
            x = self.classifier(x)
            return x


    net = CIFAR()
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    lr = lrIni
    optimizer = optim.SGD(net.parameters(), lr=lr)
    test_accuracy = []
    # train_accuracy = []
    # at the end, convert it to numpy array to save it
    modelParamsClientsOld = {}
    modelParamsClientsNew = {}
    modelGradssClients = {}
    vectrorized_gradParams_aggregated_per_group= {}
    vectrorized_gradParams_aggregated={}
    vectrorized_gradParams = {}
    vectrorized_gradParams_5 ={}
    vectrorized_gradParams_7 ={}
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    print(net)
    lcount = 0
    for param in net.parameters():
        for iClient in range(nClients):
            modelParamsClientsOld[(iClient, lcount)] = param.clone().detach()
            modelParamsClientsNew[(iClient,lcount)] = param.clone().detach()
        lcount = lcount + 1


    for iRound in range(trainTotalRounds):
        lr = lrIni

        
        lcount = 0
        for param in net.parameters():
            modelParamsClientsNew[(1, lcount)].fill_(0)
            for client4 in np.arange(nClients):
                modelParamsClientsNew[(1, lcount)] += len(train_indexes[client4])/data_size * modelParamsClientsOld[(client4, lcount)].to(device)
            lcount = lcount + 1

            
        with torch.no_grad():
            lcount = 0
            for param in net.parameters():
                param.data = modelParamsClientsNew[(1, lcount)].clone().detach()
                lcount = lcount + 1

        with torch.no_grad():
            net.eval()
            test_correct = 0.0
            with torch.no_grad():
                for i_batch in range(len(test_images)):
                    inputs = test_images[i_batch]
                    labels = test_labels[i_batch]
                    outputs = net(inputs)
                    _, preds = torch.max(outputs, 1)
                    test_correct += torch.sum(preds == labels.data)
                    test_correct = test_correct.cpu()
                round_acc = test_correct.double()/test_dataset_size
        test_accuracy.append(round_acc)
        print('Round:', (iRound + 1), 'test accuracy:', round_acc)


        for iClient in range(nClients):
            with torch.no_grad():
                lcount = 0
                for param in net.parameters():
                    param.data = modelParamsClientsNew[(1,lcount)].clone().detach()
                    lcount = lcount+1


    #################################### Epochs training ####################################

            for i_epoch in np.arange(Epochs_num):
                
                # load the model with the trained weights of the best neighbor model
                for  x_batch, y_batch in  train_clients_loaders[iClient]:

                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    net.train()
                    optimizer.zero_grad()
                    outputs = net(x_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    with torch.no_grad():
                        lcount = 0
                        for param in net.parameters():
                            gradParams = param.grad.clone().detach()
                            param.data -= lr * gradParams

                            lcount = lcount+1
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                '''Applying the attack'''
            with torch.no_grad():
                lcount = 0
                for param in net.parameters():
                    modelParamsClientsOld[(iClient, lcount)] = param.data 
                    lcount = lcount + 1

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       

    np.save('CIFAR_10_Test_accuracy_byzant_{0}_Epochs_{1}_case_{2}_attack{3}_Aggregation_{4}ـNumber_Byzantines{5}'.format(byzant, Epochs_num,case,attack,Aggregation,nByzantines), test_accuracy)

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.7)


# parser.add_argument("--connet_pro", type=float,default=0.4)
args = parser.parse_args()

main(args=args)