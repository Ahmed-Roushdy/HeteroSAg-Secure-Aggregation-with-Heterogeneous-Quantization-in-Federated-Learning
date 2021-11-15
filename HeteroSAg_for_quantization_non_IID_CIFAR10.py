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
    """ This function from FedMl library
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
    nClients = args.nClients
    trainTotalRounds = args.trainTotalRounds
    Batch_size = args.Batch_size
    Epochs_num = args.Epochs_num
    Quantization_type = args.Quantization_type # Homo, no Hetro
    Distribution = args.Distribution
    lrIni = args.lr
    K_0 = args.K_0
    K_1 = args.K_1
    K_2 = args.K_2
    K_3 = args.K_3
    K_4 = args.K_4
    alpha = args.alpha
    decreasing = args.decreasing




    device = 'cuda:7' if torch.cuda.is_available() else 'cpu'



####################################################################################

    if Quantization_type == 'Homo':
        Quantizers = K_0, K_1, K_2, K_3, K_4
    elif Quantization_type == 'no':
        K_0, K_1, K_2, K_3, K_4 = 'no', 'no', 'no', 'no', 'no'
        Quantizers = 'no'
    elif Quantization_type == 'Hetro':
        Quantizers = K_0, K_1, K_2, K_3, K_4

    else:
        print('wrong choice, please choice between (Homo,no, Hetro ')





    print('same_min',Quantization_type , 'Quantizers', Quantizers, 'Epochs_num',Epochs_num, 'lrIni',lrIni,'batch',Batch_size )   # 2, 6, 8, 10, 12'
    print(device)
####################################################################################

    seed = 1
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



#########################  dividing  users and the model into 5 groups and 5 segments  respectivly####################################

    Groups = 5
    number_per_group = nClients / Groups
    #print(number_per_group)
    group = {}
    for  i in range(5):
        group[i] = np.random.choice(int(nClients), int(number_per_group), replace=False)



    ######################### preparing the indices for the segments  #########################
    Segment_length_layer4= {}

    Segment_length_layer4[0] = 21190  #first segment end index
    Segment_length_layer4[1] = 38086+Segment_length_layer4[0] #second segment  end index
    Segment_length_layer4[2] = 98304 # #third seg   end index

    Segment_length_layer5= {}

    Segment_length_layer5[0] = 36864  #4th   end index
    Segment_length_layer5[1] = 73728 # #5th   end index



######################### Data loading and preprocessing #######################################################

    train_dataset, test_dataset = get_data_set('cifar10')
    data_size = 50000

    labels = []
    for _, label in train_dataset:
        labels.append(label)

    label_list = np.array(labels)


    train_clients_loaders, train_indexes = get_train_clients_loaders(train_dataset,nClients,Batch_size,label_list,alpha)
    test_images, test_labels,test_dataset_size = get_test_loader(test_dataset, batch_size=100, device=device)    



################## Preparing the neural netowrk  ########################

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
    # print(net)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    lr = lrIni
    optimizer = optim.SGD(net.parameters(), lr = lr)
    test_accuracy = []

#################################### initializing the models ####################################


    modelParamsClientsOld = {}   ## the local model 
    modelParamsClientsNew = {}  ## this is the global model 
    modelGradssClients = {}    ## the local gradient  
    print(net)



    lcount = 0
    for param in net.parameters():
        for iClient in range(nClients):
            modelParamsClientsOld[(iClient,lcount)] = param.clone().detach() # at the end of previous round
        modelParamsClientsNew[(1,lcount)] = param.clone().detach()
        lcount = lcount+1


    l = [ 150,300]
    lr = lrIni
    # iRound_ = 0

    for iRound in range(trainTotalRounds):
        #lr = lrIni/(1+lrIni*iRound)
        #lr = lrIni / math.sqrt(iRound + 1)
        lr = lrIni
        if iRound >150 :
            # iRound_=+1
            lr = lrIni/(1+lrIni*(iRound))        
            


#################################### global model is the average of the local models ####################################

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


#################################### compute the test  accuracy using the global model  ####################################


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
        print('decreasing', Quantization_type)



####################################  Perfrom local training ####################################


        Max_ = []
        Min_ = []

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
                            modelParamsClientsOld[(iClient, lcount)] = param.clone().detach()
                            #print('modelParamsClientsOld',torch.max(modelParamsClientsOld[(iClient, lcount)]))
                            #print('modelParamsClientsOld',torch.min(modelParamsClientsOld[(iClient, lcount)]))

                            lcount = lcount+1

########################### Store all clinets gradient and get [r1,r2], the range of the gradinet along all the clients, #####

            with torch.no_grad():
                lcount = 0
                for param in net.parameters():
                    # gradParams =Quantizer(gradParams, 2, -1, 1)
                    # modelGradssClients[(iClient, lcount)] = param.grad.clone().detach()  # at the end of previous round
                    # modelGradssClients[(iClient, lcount)] = gradParams  # at the end of previous round

                    # modelGradssClients[(iClient, lcount)] =Quantizer( modelGradssClients[(iClient, lcount)], 2 ,-1, 1)
                    # modelParamsClientsOld[(iClient, lcount)] = param.data - lr * \
                    #                                         modelGradssClients[(iClient, lcount)]
                    modelParamsClientsOld[(iClient, lcount)] = modelParamsClientsOld[(iClient, lcount)].to(device)
                    modelGradssClients[(iClient, lcount)] = modelParamsClientsOld[(iClient, lcount)] - \
                                                            modelParamsClientsNew[(1, lcount)] # get the difference between the local and global  models

                    a = torch.max(modelGradssClients[(iClient, lcount)]).to('cpu')
                    b= torch.min(modelGradssClients [(iClient, lcount)]).to('cpu')
                    #print('modelParamsClientsNew',torch.max(modelParamsClientsNew[(1, lcount)]))
                    #print('modelParamsClientsNew',torch.min(modelParamsClientsNew[(1, lcount)]))
                    time.sleep(.009)
                    

                    Max_ = np.append(Max_, a)
                    Min_ = np.append(Min_, b)
                    lcount = lcount + 1
            #print(iClient, modelGradssClients[(iClient, 1)].shape)

        max_input = np.max(Max_)
        min_input = np.min(Min_)
        print('max_input',np.isnan(max_input).any())
        print('min_input',np.isnan(min_input).any())



        #min = -1
        #max = 1
        vectrorized_gradParams_0= torch.tensor([])
        ## Loop over the stored set of gradient to do quantization and model update##########
        # with abuse of notations we consider the bias as layer
        a = [ 0, 2] # layer 0 and 2
        A = [ 5, 7, 9] #layer 5 and 7 9
        b={}
        b[0] = [16,3, 3,3] # first  convolution layer shape
        b[2] = [64,16,4,4] # second  convolution layer shape
        for iClient in range(nClients):
            for lcount_loop  in np.arange(10):
                if lcount_loop in a:  # segment 0 
                    vectrorized_gradParams_0 = torch.flatten(modelGradssClients[(iClient, lcount_loop)])

                    if iClient in group[0]:
                        gradParams_segment_0_quantized = Quantizer_1(vectrorized_gradParams_0, K_0, min_input, max_input)
                        modelGradssClients [(iClient, lcount_loop)]= torch.reshape(gradParams_segment_0_quantized, (b[lcount_loop]))
                    elif iClient in group [1]:
                        gradParams_segment_0_quantized = Quantizer_1(vectrorized_gradParams_0, K_0, min_input, max_input)
                        modelGradssClients[(iClient, lcount_loop)] = torch.reshape(gradParams_segment_0_quantized, (b[lcount_loop]))

                    elif iClient in group [2]:
                        gradParams_segment_0_quantized = Quantizer_1(vectrorized_gradParams_0, K_2, min_input, max_input)
                        modelGradssClients[(iClient, lcount_loop)] = torch.reshape(gradParams_segment_0_quantized, (b[lcount_loop]))

                    elif iClient in  group[3]:
                        gradParams_segment_0_quantized = Quantizer_1(vectrorized_gradParams_0, K_3, min_input, max_input)
                        modelGradssClients[(iClient, lcount_loop)] = torch.reshape(gradParams_segment_0_quantized, (b[lcount_loop]))
                    else:
                        gradParams_segment_0_quantized = Quantizer_1(vectrorized_gradParams_0, K_2, min_input, max_input)
                        modelGradssClients[(iClient, lcount_loop)] = torch.reshape(gradParams_segment_0_quantized,
                                                                                (b[lcount_loop]))


                elif lcount_loop == 1 : # segment 0 # bias of layer1
                        if iClient in group[0]:
                            modelGradssClients[(iClient, lcount_loop)] = Quantizer_1(modelGradssClients[(iClient, lcount_loop)], K_0, min_input, max_input)
                        elif iClient in group[1]:
                            modelGradssClients[(iClient, lcount_loop)] = Quantizer_1(modelGradssClients[(iClient, lcount_loop)], K_0, min_input, max_input)
                        elif iClient in group[2]:
                            modelGradssClients[(iClient, lcount_loop)] = Quantizer_1(modelGradssClients[(iClient, lcount_loop)], K_2, min_input, max_input)
                        elif iClient in group[3]:
                            modelGradssClients[(iClient, lcount_loop)] = Quantizer_1(modelGradssClients[(iClient, lcount_loop)], K_3, min_input, max_input)
                        else:
                            modelGradssClients[(iClient, lcount_loop)] = Quantizer_1(modelGradssClients[(iClient, lcount_loop)], K_2, min_input, max_input)

                elif lcount_loop == 4: # this layer will divided into three segment (1) one part complete segment 0 (2) two parts for two segments
                    vectrorized_gradParams_0 = torch.flatten(modelGradssClients[(iClient, lcount_loop)])

                    gradParams_segment_0 = vectrorized_gradParams_0[0:Segment_length_layer4[0]] # segment 0
                    gradParams_segment_1 = vectrorized_gradParams_0[Segment_length_layer4[0]:Segment_length_layer4[1]] # segment 1
                    gradParams_segment_2 = vectrorized_gradParams_0[Segment_length_layer4[1]:Segment_length_layer4[2]] # segment 2



                    if iClient in group[0]:
                        gradParams_segment_0_quantized = Quantizer_1(gradParams_segment_0, K_0, min_input, max_input)
                        gradParams_segment_1_quantized = Quantizer_1(gradParams_segment_1, K_0, min_input, max_input)
                        gradParams_segment_2_quantized = Quantizer_1(gradParams_segment_2, K_0, min_input, max_input)

                        vectrorized_gradParams_0 = torch.cat((gradParams_segment_0_quantized,
                                                            gradParams_segment_1_quantized,
                                                            gradParams_segment_2_quantized,))



                        modelGradssClients[(iClient, lcount_loop)] = torch.reshape(vectrorized_gradParams_0,(384, 256))
                    elif iClient in group[1]:
                        gradParams_segment_0_quantized = Quantizer_1(gradParams_segment_0, K_0, min_input, max_input)
                        gradParams_segment_1_quantized = Quantizer_1(gradParams_segment_1, K_1, min_input, max_input)
                        gradParams_segment_2_quantized = Quantizer_1(gradParams_segment_2, K_1, min_input, max_input)

                        vectrorized_gradParams_0 = torch.cat((gradParams_segment_0_quantized,
                                                            gradParams_segment_1_quantized,
                                                            gradParams_segment_2_quantized,))
                        modelGradssClients[(iClient, lcount_loop)] = torch.reshape(vectrorized_gradParams_0,
                                                                                    (384, 256))
                    elif iClient in group[2]:
                        gradParams_segment_0_quantized = Quantizer_1(gradParams_segment_0, K_2, min_input, max_input)
                        gradParams_segment_1_quantized = Quantizer_1(gradParams_segment_1, K_0, min_input, max_input)
                        gradParams_segment_2_quantized = Quantizer_1(gradParams_segment_2, K_1, min_input, max_input)

                        vectrorized_gradParams_0 = torch.cat((gradParams_segment_0_quantized,
                                                            gradParams_segment_1_quantized,
                                                            gradParams_segment_2_quantized,))
                        modelGradssClients[(iClient, lcount_loop)] = torch.reshape(vectrorized_gradParams_0,
                                                                                    (384, 256))
                    elif iClient in group[3]:
                        gradParams_segment_0_quantized = Quantizer_1(gradParams_segment_0, K_3, min_input, max_input)
                        gradParams_segment_1_quantized = Quantizer_1(gradParams_segment_1, K_3, min_input, max_input)
                        gradParams_segment_2_quantized = Quantizer_1(gradParams_segment_2, K_0, min_input, max_input)

                        vectrorized_gradParams_0 = torch.cat((gradParams_segment_0_quantized,
                                                            gradParams_segment_1_quantized,
                                                            gradParams_segment_2_quantized,))
                        modelGradssClients[(iClient, lcount_loop)] = torch.reshape(vectrorized_gradParams_0,
                                                                                    (384, 256))
                    else:
                        gradParams_segment_0_quantized = Quantizer_1(gradParams_segment_0, K_2, min_input, max_input)
                        gradParams_segment_1_quantized = Quantizer_1(gradParams_segment_1, K_3, min_input, max_input)
                        gradParams_segment_2_quantized = Quantizer_1(gradParams_segment_2, K_4, min_input, max_input)

                        vectrorized_gradParams_0 = torch.cat((gradParams_segment_0_quantized,
                                                            gradParams_segment_1_quantized,
                                                            gradParams_segment_2_quantized,))
                        modelGradssClients[(iClient, lcount_loop)] = torch.reshape(vectrorized_gradParams_0,
                                                                                    (384, 256))
                elif lcount_loop == 3 : # segment 0 Bias of the first layer
                        if iClient in group[0]:
                            modelGradssClients[(iClient, lcount_loop)] = Quantizer_1(modelGradssClients[(iClient, lcount_loop)], K_0, min_input, max_input)
                        elif iClient in group[1]:
                            modelGradssClients[(iClient, lcount_loop)] = Quantizer_1(modelGradssClients[(iClient, lcount_loop)], K_0, min_input, max_input)
                        elif iClient in group[2]:
                            modelGradssClients[(iClient, lcount_loop)] = Quantizer_1(modelGradssClients[(iClient, lcount_loop)], K_2, min_input, max_input)
                        elif iClient in group[3]:
                            modelGradssClients[(iClient, lcount_loop)] = Quantizer_1(modelGradssClients[(iClient, lcount_loop)], K_3, min_input, max_input)
                        else:
                            modelGradssClients[(iClient, lcount_loop)] = Quantizer_1(modelGradssClients[(iClient, lcount_loop)], K_2, min_input, max_input)
                if lcount_loop == 6: #segment 3 and 4
                    vectrorized_gradParams_0 = torch.flatten(modelGradssClients[(iClient, lcount_loop)])

                    gradParams_segment_3 = vectrorized_gradParams_0[0:Segment_length_layer5[0]] # segment 3
                    gradParams_segment_4 = vectrorized_gradParams_0[Segment_length_layer5[0]:Segment_length_layer5[1]] # segment 4

                    if iClient in group[0]:
                        gradParams_segment_3_quantized = Quantizer_1(gradParams_segment_3, K_0, min_input, max_input)
                        gradParams_segment_4_quantized = Quantizer_1(gradParams_segment_4, K_0, min_input, max_input)
                        # print(list(gradParams_segment_0_quantized.size()), list(gradParams_segment_1_quantized.size()), list(gradParams_segment_2_quantized.size()),list(gradParams_segment_3_quantized.size()),list(gradParams_segment_4_quantized.size()))

                        vectrorized_gradParams_0 = torch.cat((gradParams_segment_3_quantized,
                                                            gradParams_segment_4_quantized,
                                                            ))
                        modelGradssClients[(iClient, lcount_loop)] = torch.reshape(vectrorized_gradParams_0,(192, 384))
                    elif iClient in group[1]:
                        gradParams_segment_3_quantized = Quantizer_1(gradParams_segment_3, K_1, min_input, max_input)
                        gradParams_segment_4_quantized = Quantizer_1(gradParams_segment_4, K_1, min_input, max_input)
                        # print(list(gradParams_segment_0_quantized.size()), list(gradParams_segment_1_quantized.size()), list(gradParams_segment_2_quantized.size()),list(gradParams_segment_3_quantized.size()),list(gradParams_segment_4_quantized.size()))

                        vectrorized_gradParams_0 = torch.cat((gradParams_segment_3_quantized,
                                                            gradParams_segment_4_quantized,
                                                            ))
                        modelGradssClients[(iClient, lcount_loop)] = torch.reshape(vectrorized_gradParams_0, (192, 384))
                    elif iClient in group[2]:
                        gradParams_segment_3_quantized = Quantizer_1(gradParams_segment_3, K_2, min_input, max_input)
                        gradParams_segment_4_quantized = Quantizer_1(gradParams_segment_4, K_2, min_input, max_input)
                        # print(list(gradParams_segment_0_quantized.size()), list(gradParams_segment_1_quantized.size()), list(gradParams_segment_2_quantized.size()),list(gradParams_segment_3_quantized.size()),list(gradParams_segment_4_quantized.size()))

                        vectrorized_gradParams_0 = torch.cat((gradParams_segment_3_quantized,
                                                            gradParams_segment_4_quantized,
                                                            ))
                        modelGradssClients[(iClient, lcount_loop)] = torch.reshape(vectrorized_gradParams_0, (192, 384))
                    elif iClient in group[3]:
                        gradParams_segment_3_quantized = Quantizer_1(gradParams_segment_3, K_1, min_input, max_input)
                        gradParams_segment_4_quantized = Quantizer_1(gradParams_segment_4, K_2, min_input, max_input)
                        # print(list(gradParams_segment_0_quantized.size()), list(gradParams_segment_1_quantized.size()), list(gradParams_segment_2_quantized.size()),list(gradParams_segment_3_quantized.size()),list(gradParams_segment_4_quantized.size()))

                        vectrorized_gradParams_0 = torch.cat((gradParams_segment_3_quantized,
                                                            gradParams_segment_4_quantized,
                                                            ))
                        modelGradssClients[(iClient, lcount_loop)] = torch.reshape(vectrorized_gradParams_0, (192, 384))
                    else:
                        gradParams_segment_3_quantized = Quantizer_1(gradParams_segment_3, K_0, min_input, max_input)
                        gradParams_segment_4_quantized = Quantizer_1(gradParams_segment_4, K_1, min_input, max_input)
                        # print(list(gradParams_segment_0_quantized.size()), list(gradParams_segment_1_quantized.size()), list(gradParams_segment_2_quantized.size()),list(gradParams_segment_3_quantized.size()),list(gradParams_segment_4_quantized.size()))

                        vectrorized_gradParams_0 = torch.cat((gradParams_segment_3_quantized,
                                                            gradParams_segment_4_quantized,
                                                            ))
                        modelGradssClients[(iClient, lcount_loop)] = torch.reshape(vectrorized_gradParams_0, (192, 384))



                elif lcount_loop in A: # segment 4
                    if iClient in group[0]:
                        modelGradssClients[(iClient, lcount_loop)] = Quantizer_1(modelGradssClients[(iClient, lcount_loop)],
                                                                                K_0, min_input, max_input)
                    elif iClient in group[1]:
                        modelGradssClients[(iClient, lcount_loop)] = Quantizer_1(modelGradssClients[(iClient, lcount_loop)],
                                                                                K_1, min_input, max_input)
                    elif iClient in group[2]:
                        modelGradssClients[(iClient, lcount_loop)] = Quantizer_1(modelGradssClients[(iClient, lcount_loop)],
                                                                                K_2, min_input, max_input)
                    elif iClient in group[3]:
                        modelGradssClients[(iClient, lcount_loop)] = Quantizer_1(modelGradssClients[(iClient, lcount_loop)],
                                                                                K_2, min_input, max_input)
                    else:
                        modelGradssClients[(iClient, lcount_loop)] = Quantizer_1(modelGradssClients[(iClient, lcount_loop)],
                                                                                K_1, min_input, max_input)

                b[8]= [10,192]
                if lcount_loop == 8:  # segment 4

                    #print(modelGradssClients [(iClient, lcount_loop)].shape)
                    vectrorized_gradParams_0 = torch.flatten(modelGradssClients[(iClient, lcount_loop)])

                    if iClient in group[0]:
                        gradParams_segment_0_quantized = Quantizer_1(vectrorized_gradParams_0, K_0, min_input, max_input)

                        modelGradssClients [(iClient, lcount_loop)]= torch.reshape(gradParams_segment_0_quantized, (b[lcount_loop]))
                    elif iClient in group[1]:
                        gradParams_segment_0_quantized = Quantizer_1(vectrorized_gradParams_0, K_1, min_input, max_input)
                        modelGradssClients[(iClient, lcount_loop)] = torch.reshape(gradParams_segment_0_quantized, (b[lcount_loop]))

                    elif iClient in group[2]:
                        gradParams_segment_0_quantized = Quantizer_1(vectrorized_gradParams_0, K_2, min_input, max_input)
                        modelGradssClients[(iClient, lcount_loop)] = torch.reshape(gradParams_segment_0_quantized, (b[lcount_loop]))

                    elif iClient in group[3]:
                        gradParams_segment_0_quantized = Quantizer_1(vectrorized_gradParams_0, K_2, min_input, max_input)
                        modelGradssClients[(iClient, lcount_loop)] = torch.reshape(gradParams_segment_0_quantized, (b[lcount_loop]))
                    else:
                        gradParams_segment_0_quantized = Quantizer_1(vectrorized_gradParams_0, K_1, min_input, max_input)
                        modelGradssClients[(iClient, lcount_loop)] = torch.reshape(gradParams_segment_0_quantized,
                                                                                (b[lcount_loop]))

               
                # #####  modle grad is the model update& we add the global model to it to have a correct  global model when averaging (equation-wise)
                
                modelParamsClientsOld[(iClient, lcount_loop)] = modelGradssClients[(iClient, lcount_loop)].to('cpu')+ modelParamsClientsNew[(1, lcount_loop)].to('cpu')
        #lrIni = .99*lrIni
    np.save(
        'CIFAR10_Test_accuracy_Quatization_type_{0}_Batch_size_{1}_Quantizers_{2}_Epochs_{3}_learning_rate_{4}_K_{5}_K_{6}_K_{7}_K_{8}_K_{9}_alpha_{10}_decreasing_{11}_rounds_{12}'.format(
            Quantization_type, Batch_size, Quantizers, Epochs_num,  lrIni, K_0, K_1, K_2, K_3, K_4, alpha,decreasing_trainTotalRounds ), test_accuracy)

    print(K_0,K_1,K_2,K_3,K_4)
        
parser = argparse.ArgumentParser()
parser.add_argument("--nClients", type=int, default=25)
parser.add_argument("--trainTotalRounds", type=int, default=500)
parser.add_argument("--Batch_size", type=float, default=0.2)
parser.add_argument("--Epochs_num", type=int)
parser.add_argument("--Quantization_type",type=str)
parser.add_argument("--Distribution",type=str)
parser.add_argument("--lr", type=float, default=0.06)
parser.add_argument("--K_0", type=int)
parser.add_argument("--K_1", type=int)
parser.add_argument("--K_2", type=int)
parser.add_argument("--K_3", type=int)
parser.add_argument("--K_4", type=int)
parser.add_argument("--alpha", type=float, default=0.7)
parser.add_argument("--decreasing", type=int, default=0)




# parser.add_argument("--connet_pro", type=float,default=0.4)
args = parser.parse_args()

main(args=args)














