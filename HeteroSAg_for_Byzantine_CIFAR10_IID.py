##################################### import ###################################
import torch
import math
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
import os
import h5py
from sklearn.utils import shuffle
from math import ceil
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import time
import random
import argparse

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



def main(args):
    case1 = args.case1
    case= args.case

    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'


    mean1 = 0.4914
    mean2 = 0.4822
    mean3 = 0.4465
    SD1 = 0.2023
    SD2 = 0.1994
    SD3 = 0.2010
    epochs = 10
    nClasses = 10
    nClients = 300
    trainTotalRounds = 300
    lrIni = 0.06
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
    #Generate segment indices, i.e., segments_3 = [ 0, 1, 2,  3 ,4] mean that we use the varibale  segments_3 to point to thses segment indices [ 0, 1, 2,  3 ,4]
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
    # collected groups generate the groups that should enocd and decode segments for each level together, Example 2 in the paper (0,1), 2, (3,4) over the first segment
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
    non_iid = 0
    case = 4
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

    case1 = 2
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
    #Load and proccess the CIFAR10 dataset
    xtr = np.load('cifar10-xTrainSorted.npy')
    ytr = np.load('cifar10-yTrainSorted.npy')
    trainDataSize = np.size(xtr, 0)
    trainDataSize = np.size(xtr, 0)
    normalMeanTrain = np.zeros(np.shape(xtr))
    normalStdTrain = np.zeros(np.shape(xtr))
    normalMeanTrain[:, 0:1024] = mean1
    normalMeanTrain[:, 1024:2048] = mean2
    normalMeanTrain[:, 2048:3072] = mean3
    normalStdTrain[:, 0:1024] = SD1
    normalStdTrain[:, 1024:2048] = SD2
    normalStdTrain[:, 2048:3072] = SD3
    xte = np.load('cifar10-xTest.npy')
    yte = np.load('cifar10-yTest.npy')
    testDataSize = np.size(xte, 0)
    normalMeanTest = np.zeros(np.shape(xte))
    normalStdTest = np.zeros(np.shape(xte))
    normalMeanTest[:, 0:1024] = mean1
    normalMeanTest[:, 1024:2048] = mean2
    normalMeanTest[:, 2048:3072] = mean3
    normalStdTest[:, 0:1024] = SD1
    normalStdTest[:, 1024:2048] = SD2
    normalStdTest[:, 2048:3072] = SD3
    xtr = xtr / 255.0
    xtr = (xtr - normalMeanTrain) / normalStdTrain
    xtr = np.float32(xtr)
    xtr = np.reshape(xtr, (trainDataSize, 3, 32, 32))
    ytr = np.int64(ytr)
    xtr = torch.from_numpy(xtr)
    ytr = torch.from_numpy(ytr)
    xtr = xtr.to(device)
    ytr = ytr.to(device)

    xte = xte / 255.0
    xte = (xte - normalMeanTest) / normalStdTest
    xte = np.float32(xte)
    xte = np.reshape(xte, (testDataSize, 3, 32, 32))
    yte = np.int64(yte)
    xte = torch.from_numpy(xte)
    yte = torch.from_numpy(yte)

    xte = xte.to(device)
    yte = yte.to(device)


    temp0 = np.random.permutation(trainDataSize)
    xtr_0 = xtr[temp0]
    ytr_0 = ytr[temp0]
    number_of_rows = np.arange(50000)



    trainDataSizeFracClients = 1/nClients

    trainDataSizeClients = np.int32(trainDataSizeFracClients * trainDataSize)

    ###
    if non_iid == 0:
        temp1 = np.random.permutation(trainDataSize)
        xtr = xtr[temp1]
        ytr = ytr[temp1]

    print('some y samples:', ytr[0:20])
    print('here',byzantClientsList)
    trainDataSizeFracClients = np.array([1 for iClient3 in range(nClients)])
    trainDataSizeFracClients = trainDataSizeFracClients / np.sum(trainDataSizeFracClients)
    print('trainDataSizeClients:', trainDataSizeClients)
    # time.sleep(3)
    miniBatchSizeFrac = np.array([0.2 for iClient in range(nClients)])
    miniBatchSizeClients = np.int32(miniBatchSizeFrac * trainDataSizeClients)
    clientsEpochs = np.array([ epochs for iClient in range(nClients)])
    nMiniBatchesClientsPerEpoch = np.int32(trainDataSizeClients / miniBatchSizeClients)

    trainDataIndices = {}
    stIndex = 0
    for iClient in range(nClients):
        trainDataIndices[(iClient, 0)] = stIndex
        trainDataIndices[(iClient, 1)] = (stIndex + trainDataSizeClients)
        stIndex = (stIndex + trainDataSizeClients)
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
            xTestBatch = xte
            yTestBatch = yte
            outputs = net(xTestBatch)
            _, predicted_val = torch.max(outputs.data, 1)
            total = yTestBatch.size(0)
            correct = (predicted_val == yTestBatch).sum().item()
            accIter = correct / total
        test_accuracy.append(accIter)
        print('Round:', (iRound + 1), 'test accuracy:', accIter)

        for iClient in range(nClients):
            with torch.no_grad():
                lcount = 0
                for param in net.parameters():
                    param.data = modelParamsClientsNew[(iClient, lcount)].clone().detach()
                    lcount = lcount + 1
            clientDataX = xtr[trainDataIndices[(iClient, 0)]:trainDataIndices[(iClient, 1)]]
            clientDataY_original = ytr[trainDataIndices[(iClient, 0)]:trainDataIndices[(iClient, 1)]]
            if case == 4:
                if iClient in byzantClientsList:
                    clientDataY =  9- ytr[trainDataIndices[(iClient, 0)]:trainDataIndices[(iClient, 1)]]
                else:
                    clientDataY = ytr[trainDataIndices[(iClient, 0)]:trainDataIndices[(iClient, 1)]]
            else:
                clientDataY = ytr[trainDataIndices[(iClient, 0)]:trainDataIndices[(iClient, 1)]]
            seed = iRound * nClients + iClient
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            nEpochs = clientsEpochs[iClient]
            nBatches = nMiniBatchesClientsPerEpoch[iClient]
            miniBatchSize = miniBatchSizeClients[iClient]
            for iEpoch in range(nEpochs):
                for iTrainMiniBatch in range(nBatches):
                    xbatch = clientDataX[iTrainMiniBatch * miniBatchSize:(iTrainMiniBatch + 1) * miniBatchSize]
                    ybatch = clientDataY[iTrainMiniBatch * miniBatchSize:(iTrainMiniBatch + 1) * miniBatchSize]
                    net.train()
                    optimizer.zero_grad()
                    outputs = net(xbatch)
                    loss = criterion(outputs, ybatch)
                    loss.backward()
                    with torch.no_grad():
                        lcount = 0
                        for param in net.parameters():
                            gradParams = param.grad.clone().detach()
                            param.data -= lr * gradParams
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            with torch.no_grad():
                lcount = 0
                for param in net.parameters():
                    modelGradssClients[(iClient, lcount)] = param.grad.clone().detach()
                    modelParamsClientsOld[(iClient, lcount)] = param.data - lr * \
                                                            modelGradssClients[(iClient, lcount)]
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

                            # sum the  gradient from users in each group (can be visulaised as the SS matrix )
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
                    for iClient in Group[group_index]:
                        if (lcount_loop in small_layer  and  segment_index ==  Groups-1) :
                            vectrorized_gradParams_aggregated_per_group[(lcount_loop,segment_index, group_index)]+=1/number_per_group*vectrorized_gradParams[(lcount_loop, segment_index,np.int(iClient), group_index)].to('cpu')



                        elif (lcount_loop == 4 and segment_index in segments_5):
                            vectrorized_gradParams_aggregated_per_group[(lcount_loop, segment_index, group_index)]+=1/number_per_group*vectrorized_gradParams[(lcount_loop, segment_index, np.int(iClient), group_index)].to('cpu')
                        elif (lcount_loop == 2 and segment_index in segments_3):
                            vectrorized_gradParams_aggregated_per_group[
                            (lcount_loop, segment_index, group_index)] += 1 / number_per_group * vectrorized_gradParams[
                            (lcount_loop, segment_index, np.int(iClient), group_index)].to('cpu')

                        elif (lcount_loop == 6 and segment_index in segments_7):
                            vectrorized_gradParams_aggregated_per_group[
                            (lcount_loop, segment_index, group_index)] += 1 / number_per_group * vectrorized_gradParams[
                            (lcount_loop, segment_index, np.int(iClient), group_index)].to('cpu')
                            # intialize the dictionary where we put the  gradients average of  groups that have the same index in the SS matrix
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
                    ### compute median


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




    np.save('CIFAR_10_Test_accuracy_byzant_{0}_Epochs_{1}__non_iid_{2}_case_{3}_attack{4}_Aggregation_{5}ـNumber_Byzantines{6}'.format(byzant, epochs,  non_iid,case,attack,Aggregation,nByzantines), test_accuracy)

parser = argparse.ArgumentParser()
parser.add_argument("--case", type=int, default=1)  # 2:Gaussian, 3:sign flip
parser.add_argument("--case1", type=int, default=1) # 1:mean, 2:median


# parser.add_argument("--connet_pro", type=float,default=0.4)
args = parser.parse_args()

main(args=args)