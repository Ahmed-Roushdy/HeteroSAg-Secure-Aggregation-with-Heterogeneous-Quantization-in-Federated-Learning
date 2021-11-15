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
    case1 = args.case1
    case= args.case
    alpha = args.alpha



    device = 'cuda:7' if torch.cuda.is_available() else 'cpu'







    Epochs_num = 4
    nClasses = 10
    nClients = 300
    trainTotalRounds = 300
    Batch_size = 0.2
    lrIni = 0.03
    lrFac = 1.0
    Groups = 75
    layer = 10 # the total number of weights  as tensors  and biases   ( Each layer   is counted as 2)
    number_per_group = np.int32(nClients / Groups) # number of user in each group



    seed = 1
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Segment_0_length = total number of parameter over G


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Group= {} # dictionary has a key as the group index and value is the users indices for this group
    i = 0
    J= np.zeros(number_per_group)
    for group in range(Groups):
        Group[group] = np.arange(number_per_group) + i + J[-1]
        i =1
        J = Group[group]

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''Generate segment indices: We have 75 segment. 
    In the following  segments_3 = [ 0, 1, 2,  3 ,4] mean that we use the varibale  segments_3 to point to thses segment indices [ 0, 1, 2,  3 ,4]'''
    i = 0
    segments_3 = [ 0, 1, 2, 3 ,4]
    q = range(5,45)
    segments_5 = list(q)
    print(segments_5)

    q2= range(45,45+29)
    segments_7 = list(q2)
    print(segments_7)

    # Pointer for layer 3 (weights only)
    Segment_length_layer3={}
    i = 0
    layer_3_param_size = 64*16*16
    print(layer_3_param_size)
    param_per_group_layer_3 = np.int32(layer_3_param_size /6)
    #generate pointer for the staritng point and the end point for each segment resulted from segmenting the layer_3 weights
    for seg in segments_3:
        Segment_length_layer3[seg] = param_per_group_layer_3 + i
        i = Segment_length_layer3[seg]
    Segment_length_layer3[4] = 16384


    Segment_length_layer5={}
    i = 0
    layer_5_param_size = 384*256
    print(layer_5_param_size)
    param_per_group_layer_5 = np.int32(layer_5_param_size /40)
    #generate pointer for the staritng point and the end point for each segment resulted from segmenting the layer_5 weights FC layer 384*256
    for seg in segments_5:
        Segment_length_layer5[seg] = param_per_group_layer_5 + i
        i = Segment_length_layer5[seg]
        #print(Segment_length_layer5[seg])
    Segment_length_layer5[44] = 98304


    # Pointer for layer 7
    Segment_length_layer7={}
    i = 0
    layer_7_param_size = 384*192
    print(layer_7_param_size)
    param_per_group_layer_7 = np.int32(layer_7_param_size /29)
    for seg in segments_7:
        Segment_length_layer7[seg] = param_per_group_layer_7 + i
        i = Segment_length_layer7[seg]
    Segment_length_layer7[73] = 73728

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Generate the SS selection matrix for HetroSAg
    B = np.zeros([Groups,Groups])+1000
    for g in np.arange(Groups -1):
        for r in np.arange(Groups -1-g):
            l = 2*g+r
            B [l%Groups][g] =g
            B [l%Groups][g+r+1]  = g
    #print(B)
    # Grouped set of users per each segments  segment [0] 0 0 1 * 1
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # collected groups:  generate the groups that should enocd and decode segments for each level together, Example 2 in the paper (0,1), 2, (3,4) over the first segment
    Collected_groups = {}
    for segment in np.arange(Groups):
        i = 0
        for column_index in np.arange(Groups):
            if B[segment][column_index] == 1000:
                Collected_groups[(segment, i)] = column_index
                i = i + 1
                # print('1000', Collected_groups)
            else:
                for scanned_column in range(column_index + 1, Groups):
                    # print(scanned_column)
                    if (B[segment][column_index] == B[segment][scanned_column]) and (column_index != scanned_column):
                        Collected_groups[(segment, i)] = np.array([column_index, scanned_column])
                        # print('2', Collected_groups)
                        i = i + 1

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #The  differnet  simulations cases we have
    # if attack = 'No_attack':
    #     byzant = 0
    # elif attack = 'gaussian':
    #     byzant = 1
    # elif attack = 'sign_flip':
    #     attack = 'sign_flip'
    #     byzant = 1
    # else:
    #     attack = 'label_flip'
    #     byzant = 1


    # if aggregation == ' mean':
    #     Aggregation = ' mean'
    # elif Aggregation = 'median':
    #     Aggregation = 'median'


    #The  differnet  simulations cases we have
    if case == 1:
        byzant = 0
        attack = 'No-attack'
    elif case == 2:
        attack = 'gaussian'
        byzant = 1
    elif case == 3:
        attack = 'sign_flip'
        byzant = 1
    else:
        attack = 'label_flip'
        byzant = 1

    if case1 == 1:
        Aggregation = ' mean'
    elif case1 == 2:
        Aggregation = 'median'
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Generating the Byzantine list
    j =1
    byzantClientsList = np.array([])
    nByzantines = 18 # agenda 1 for no byzantines
    print('_______')
    for i in range(nByzantines):
        j = 4*i
        byzantClientsList =  np.append(byzantClientsList, j)
    print('here',byzantClientsList)
    print("--------")
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
        for iClient in range(nClients):  # The average old model will be copied for each client
            for lcount in np.arange(layer):
                modelParamsClientsNew[(iClient, lcount)] =  modelParamsClientsOld[(iClient, lcount)]

        with torch.no_grad():
            lcount = 0
            for param in net.parameters():
                param.data = modelParamsClientsNew[(2, lcount)].clone().detach()
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
                    modelGradssClients[(iClient, lcount)] = modelParamsClientsOld[(iClient, lcount)] - \
                                                            modelParamsClientsNew[(iClient, lcount)]
                    if byzant == 1:
                        if iClient in byzantClientsList:
                            if attack =='gaussian':
                                modelGradssClients[(iClient, lcount)] = 5* torch.randn(np.shape(param))
                            elif attack =='sign_flip':
                                modelGradssClients[(iClient, lcount)] = -1 * torch.randn(np.shape(param))
                            elif attack =='label_flip':
                                modelGradssClients[(iClient, lcount)] = 30 * modelGradssClients[(iClient, lcount)]
                        else:
                            modelGradssClients[(iClient, lcount)] = modelGradssClients[(iClient, lcount)]
                    else:
                        modelGradssClients[(iClient, lcount)] = modelGradssClients[(iClient, lcount)]
                    lcount = lcount + 1

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        ''' In the following  Generating a disctionary with the  keys: the layer index: we abuse the notation and condier the bias as a layer"
                                                    segment index: eache  layer is divided into a set of segments
                                                    iClient and group_index that clients belongs
                                                    at the end we will have G=75 segments with almost the same number of elemnts

        '''
        for iClient in range(nClients):
            for lcount_loop in np.arange(layer):
                if lcount_loop == 2:
                    for group_index in np.arange(Groups):
                        if iClient in Group[group_index]:
                            vectrorized_gradParams_0 = torch.flatten(modelGradssClients[(iClient, lcount_loop)])
                            i=0
                            for segment_index in segments_3:
                                vectrorized_gradParams[( lcount_loop,segment_index, iClient, group_index)] = vectrorized_gradParams_0[i:Segment_length_layer3[segment_index]]
                                i = Segment_length_layer3[segment_index]

                if lcount_loop == 4:
                    for group_index in np.arange(Groups):
                        if iClient in Group[group_index]:
                            vectrorized_gradParams_5 = torch.flatten(modelGradssClients[(iClient, lcount_loop)])
                            i=0
                            for segment_index in segments_5:
                                vectrorized_gradParams[( lcount_loop,segment_index, iClient, group_index)] = vectrorized_gradParams_5[i:Segment_length_layer5[segment_index]]
                                i = Segment_length_layer5[segment_index]

                if lcount_loop == 6:
                    for group_index in np.arange(Groups):
                        if iClient in Group[group_index]:
                            vectrorized_gradParams_7 = torch.flatten(modelGradssClients[(iClient, lcount_loop)])
                            i=0
                            for segment_index in segments_7:
                                vectrorized_gradParams[( lcount_loop,segment_index, iClient, group_index)] = vectrorized_gradParams_7[i:Segment_length_layer7[segment_index]]
                                i = Segment_length_layer7[segment_index]
                if lcount_loop == 1:
                    for group_index in np.arange(Groups):
                        if iClient in Group[group_index]:
                            segment_index =  Groups-1
                            vectrorized_gradParams[(lcount_loop, segment_index, iClient, group_index)] = modelGradssClients[(iClient, lcount_loop)]

                if lcount_loop == 0:
                    for group_index in np.arange(Groups):
                        if iClient in Group[group_index]:
                            segment_index=  Groups-1
                            vectrorized_gradParams[(lcount_loop, segment_index, iClient, group_index)] = torch.flatten(modelGradssClients[(iClient, lcount_loop)])

                if lcount_loop == 3:
                    for group_index in np.arange(Groups):
                        if iClient in Group[group_index]:
                            segment_index = Groups-1

                            vectrorized_gradParams[(lcount_loop, segment_index, iClient, group_index)] = modelGradssClients[(iClient, lcount_loop)]

                if lcount_loop == 5:
                    for group_index in np.arange(Groups):
                        if iClient in Group[group_index]:
                            segment_index = Groups - 1

                            vectrorized_gradParams[(lcount_loop, segment_index, iClient, group_index)] = \
                            modelGradssClients[(iClient, lcount_loop)]

                if lcount_loop == 7:
                    for group_index in np.arange(Groups):
                        if iClient in Group[group_index]:
                            segment_index = Groups - 1

                            vectrorized_gradParams[(lcount_loop, segment_index, iClient, group_index)] = \
                            modelGradssClients[(iClient, lcount_loop)]

                if lcount_loop == 8:
                    for group_index in np.arange(Groups):
                        if iClient in Group[group_index]:
                            segment_index = Groups - 1
                            vectrorized_gradParams[(lcount_loop, segment_index, iClient, group_index)] = torch.flatten(modelGradssClients[(iClient, lcount_loop)])

                if lcount_loop == 9:
                    for group_index in np.arange(Groups):
                        if iClient in Group[group_index]:
                            segment_index = Groups - 1
                            vectrorized_gradParams[(lcount_loop, segment_index, iClient, group_index)] = \
                                modelGradssClients[(iClient, lcount_loop)]

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            ''' sum the  gradient from users in each group (can be visulaised as the SS matrix )
                            We start be intializing the aggregator directory by zeros. We will accumulate the gradients 
                            from the users in those dictionaries
                            '''
        small_layer = [0 ,1 ,3, 5, 7, 8, 9]
        large_layer = [ 2, 4, 6]
        for lcount_loop in np.arange(layer):
            for segment_index in np.arange(Groups):
                for group_index in np.arange(Groups):
                    for iClient in Group[group_index]:
                        if (lcount_loop in small_layer and segment_index ==  Groups-1) :
                            vectrorized_gradParams_aggregated_per_group[(lcount_loop,segment_index, group_index)]=torch.zeros(( vectrorized_gradParams[(lcount_loop, segment_index,np.int(iClient), group_index)].size()))
                        elif (lcount_loop == 2 and  segment_index in segments_3 ) :
                            vectrorized_gradParams_aggregated_per_group[(lcount_loop,segment_index, group_index)]=torch.zeros(( vectrorized_gradParams[(lcount_loop, segment_index,np.int(iClient), group_index)].size()))
                        elif (lcount_loop == 4 and segment_index in segments_5):
                            vectrorized_gradParams_aggregated_per_group[
                            (lcount_loop, segment_index, group_index)] = torch.zeros(
                            (vectrorized_gradParams[(lcount_loop, segment_index, np.int(iClient), group_index)].size()))
                        elif (lcount_loop == 6 and segment_index in segments_7):
                            vectrorized_gradParams_aggregated_per_group[
                            (lcount_loop, segment_index, group_index)] = torch.zeros(
                            (vectrorized_gradParams[(lcount_loop, segment_index, np.int(iClient), group_index)].size()))





    ############take the average over each group:: output G*G dictionary where each column is the average of one group
        for lcount_loop in np.arange(layer):
            for segment_index in np.arange(Groups):
                for group_index in np.arange(Groups):
                    data_size_per_group = 0
                    for iClient in Group[group_index]:
                        data_size_per_group += len(train_indexes[iClient])

                    for iClient in Group[group_index]:
                        if (lcount_loop in small_layer  and  segment_index ==  Groups-1) :
                            vectrorized_gradParams_aggregated_per_group[(lcount_loop,segment_index, group_index)]+=len(train_indexes[iClient])/data_size_per_group*vectrorized_gradParams[(lcount_loop, segment_index,np.int(iClient), group_index)].to('cpu')



                        elif (lcount_loop == 4 and segment_index in segments_5):
                            vectrorized_gradParams_aggregated_per_group[(lcount_loop, segment_index, group_index)]+=len(train_indexes[iClient])/data_size_per_group*vectrorized_gradParams[(lcount_loop, segment_index, np.int(iClient), group_index)].to('cpu')
                        elif (lcount_loop == 2 and segment_index in segments_3):
                            vectrorized_gradParams_aggregated_per_group[
                            (lcount_loop, segment_index, group_index)] += len(train_indexes[iClient])/data_size_per_group * vectrorized_gradParams[
                            (lcount_loop, segment_index, np.int(iClient), group_index)].to('cpu')

                        elif (lcount_loop == 6 and segment_index in segments_7):
                            vectrorized_gradParams_aggregated_per_group[
                            (lcount_loop, segment_index, group_index)] += len(train_indexes[iClient])/data_size_per_group * vectrorized_gradParams[
                            (lcount_loop, segment_index, np.int(iClient), group_index)].to('cpu')
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            ''' intialize the dictionary where we put the  gradients average of  groups that have 
                            the same index in the SS matrix, e..g, we will average the first segment from group 0 with the first segment from group 1'''
        for lcount_loop in np.arange(layer):
            for segment_index in np.arange(Groups):
                for merged_groups_index in np.arange((Groups+1)/2): # number of merged  when groups is odd
                    if lcount_loop in small_layer and segment_index ==  Groups-1:
                        vectrorized_gradParams_aggregated[(lcount_loop, segment_index, merged_groups_index)] = torch.zeros(
                            vectrorized_gradParams_aggregated_per_group[(lcount_loop, segment_index, 0)].size())
                    elif (lcount_loop == 2 and segment_index in segments_3):
                        vectrorized_gradParams_aggregated[(lcount_loop, segment_index, merged_groups_index)] = torch.zeros(
                            vectrorized_gradParams_aggregated_per_group[(lcount_loop, segment_index, 0)].size())
                    elif (lcount_loop == 4 and segment_index in segments_5):
                        vectrorized_gradParams_aggregated[(lcount_loop, segment_index, merged_groups_index)] = torch.zeros(
                            vectrorized_gradParams_aggregated_per_group[(lcount_loop, segment_index, 0)].size())
                    elif (lcount_loop == 6 and segment_index in segments_7):
                        vectrorized_gradParams_aggregated[(lcount_loop, segment_index, merged_groups_index)] = torch.zeros(
                            vectrorized_gradParams_aggregated_per_group[(lcount_loop, segment_index, 0)].size())


                ### compute   gradients average of  groups that have the same index in the SS matrix
        for lcount_loop in np.arange(layer):
            for segment_index in np.arange(Groups):
                for merged_groups_index in np.arange((Groups+1)/2): # number of merged  when groups is odd
                    if lcount_loop in  small_layer and segment_index  == Groups-1 :
                        if np.size(np.array([Collected_groups[(segment_index,merged_groups_index)]]))==2: # since Collected_groups either have two groups or an individual group has *
                            for group_index in  Collected_groups[(segment_index,merged_groups_index)]:
                                vectrorized_gradParams_aggregated[(lcount_loop, segment_index, merged_groups_index)]+= 0.5* vectrorized_gradParams_aggregated_per_group[  (lcount_loop,segment_index,group_index)]


                        else:
                            Bufer = Collected_groups[(segment_index, merged_groups_index)]
                            vectrorized_gradParams_aggregated[(lcount_loop, segment_index, merged_groups_index)]= vectrorized_gradParams_aggregated_per_group[(lcount_loop, segment_index,Bufer)]
                    elif (lcount_loop == 2 and segment_index in segments_3):
                        if np.size(np.array([Collected_groups[(segment_index, merged_groups_index)]])) == 2:
                            for group_index in Collected_groups[(segment_index, merged_groups_index)]:
                                vectrorized_gradParams_aggregated[(lcount_loop, segment_index, merged_groups_index)]+= 0.5*  vectrorized_gradParams_aggregated_per_group[(lcount_loop, segment_index,group_index)]
                        else:
                            Bufer = Collected_groups[(segment_index, merged_groups_index)]
                            vectrorized_gradParams_aggregated[(lcount_loop, segment_index, merged_groups_index)] = vectrorized_gradParams_aggregated_per_group[(lcount_loop, segment_index, Bufer)]
                    elif (lcount_loop == 4 and segment_index in segments_5):
                        if np.size(np.array([Collected_groups[(segment_index, merged_groups_index)]])) == 2:
                            for group_index in Collected_groups[(segment_index, merged_groups_index)]:

                                vectrorized_gradParams_aggregated[(lcount_loop, segment_index, merged_groups_index)]+= 0.5*  vectrorized_gradParams_aggregated_per_group[(lcount_loop, segment_index,group_index)]
                        else:
                            Bufer = Collected_groups[(segment_index, merged_groups_index)]
                            vectrorized_gradParams_aggregated[(lcount_loop, segment_index, merged_groups_index)] = vectrorized_gradParams_aggregated_per_group[(lcount_loop, segment_index, Bufer)]
                    elif (lcount_loop == 6 and segment_index in segments_7):
                        if np.size(np.array([Collected_groups[(segment_index, merged_groups_index)]])) == 2:
                            for group_index in Collected_groups[(segment_index, merged_groups_index)]:
                                vectrorized_gradParams_aggregated[(lcount_loop, segment_index, merged_groups_index)]+= 0.5*  vectrorized_gradParams_aggregated_per_group[(lcount_loop, segment_index,group_index)]
                        else:
                            Bufer = Collected_groups[(segment_index, merged_groups_index)]
                            vectrorized_gradParams_aggregated[(lcount_loop, segment_index, merged_groups_index)] = vectrorized_gradParams_aggregated_per_group[(lcount_loop, segment_index, Bufer)]
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            ''' we compute the median'''

        median_grad ={}
        modelGradssClients_layer={}
        for lcount_loop in np.arange(layer):
            for segment_index in np.arange(Groups):
                if lcount_loop in small_layer and segment_index == Groups-1:
                    x = [vectrorized_gradParams_aggregated[(lcount_loop,segment_index, merged_groups_index)].numpy() for merged_groups_index in np.arange((Groups+1)/2)]
                    if case1 == 1:
                        median_grad[(lcount_loop, segment_index)] = np.mean(x, axis=0)
                    else:
                        median_grad[(lcount_loop, segment_index)] = np.median(x, axis=0)

                elif (lcount_loop == 6 and segment_index in segments_7):
                    x = [vectrorized_gradParams_aggregated[(lcount_loop, segment_index, merged_groups_index)].numpy() for merged_groups_index in np.arange((Groups + 1) / 2)]
                    if case1 == 1:
                        median_grad[(lcount_loop, segment_index)] = np.mean(x, axis=0)
                    else:
                        median_grad[(lcount_loop, segment_index)] = np.median(x, axis=0)
                elif (lcount_loop == 4 and segment_index in segments_5):
                    x = [vectrorized_gradParams_aggregated[(lcount_loop, segment_index, merged_groups_index)].numpy() for
                        merged_groups_index in np.arange((Groups + 1) / 2)]
                    if case1 == 1:
                        median_grad[(lcount_loop, segment_index)] = np.mean(x, axis=0)
                    else:
                        median_grad[(lcount_loop, segment_index)] = np.median(x, axis=0)
                elif (lcount_loop == 2 and segment_index in segments_3):
                    x = [vectrorized_gradParams_aggregated[(lcount_loop, segment_index, merged_groups_index)].numpy() for
                        merged_groups_index in np.arange((Groups + 1) / 2)]
                    if case1 == 1:
                        median_grad[(lcount_loop, segment_index)] = np.mean(x, axis=0)
                    else:
                        median_grad[(lcount_loop, segment_index)] = np.median(x, axis=0)



        layer_one_dim = [1, 3, 5, 7, 9]

        for lcount_loop in np.arange(layer):

            if lcount_loop ==0 :

                modelGradssClients_layer[lcount_loop] = median_grad[(lcount_loop,  Groups-1)]
                modelGradssClients_layer[lcount_loop] = np.reshape(modelGradssClients_layer[lcount_loop], (16,3, 3, 3))
            elif lcount_loop in layer_one_dim:
                modelGradssClients_layer[lcount_loop]= median_grad[(lcount_loop,  Groups-1)]

            elif lcount_loop == 8:
                modelGradssClients_layer[lcount_loop] = median_grad[(lcount_loop,  Groups-1)]
                modelGradssClients_layer[lcount_loop] = np.reshape(modelGradssClients_layer[lcount_loop], (10,192))
            elif lcount_loop ==2 :

                modelGradssClients_layer[lcount_loop] = np.concatenate(([median_grad[(lcount_loop, segment_index)] for segment_index in segments_3]), axis = 0)


                modelGradssClients_layer[lcount_loop] = np.reshape(modelGradssClients_layer[lcount_loop], (64,16, 4, 4))
            elif lcount_loop == 4:

                modelGradssClients_layer[lcount_loop] = np.concatenate(
                ([median_grad[(lcount_loop, segment_index)] for segment_index in segments_5]), axis=0)


                modelGradssClients_layer[lcount_loop] = np.reshape(modelGradssClients_layer[lcount_loop], (384, 256))
            elif lcount_loop == 6:

                modelGradssClients_layer[lcount_loop] = np.concatenate(
                ([median_grad[(lcount_loop, segment_index)] for segment_index in  segments_7]), axis=0)


                modelGradssClients_layer[lcount_loop] = np.reshape(modelGradssClients_layer[lcount_loop], (192, 384))
        for lcount in np.arange(layer):
            modelGradssClients_layer[lcount] = torch.from_numpy(modelGradssClients_layer[lcount])
            for iClient in np.arange(nClients)   :
                modelParamsClientsOld[(iClient, lcount)]=modelParamsClientsNew[(iClient, lcount)].to(device)+ modelGradssClients_layer[lcount].to(device)




    np.save('CIFAR_10_Test_accuracy_byzant_{0}_Epochs_{1}_case_{2}_attack{3}_Aggregation_{4}ـNumber_Byzantines{5}_alpha_{6}'.format(byzant, Epochs_num,case,attack,Aggregation,nByzantines,alpha), test_accuracy)

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.7)
parser.add_argument("--case", type=int, default=1)  # 2:Gaussian, 3:sign flip
parser.add_argument("--case1", type=int, default=1) # 1:mean, 2:median


# parser.add_argument("--connet_pro", type=float,default=0.4)
args = parser.parse_args()

main(args=args)