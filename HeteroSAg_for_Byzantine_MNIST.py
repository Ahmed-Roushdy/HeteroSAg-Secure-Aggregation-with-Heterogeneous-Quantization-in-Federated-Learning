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
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
    layer_0 = 78400
    Total_param = 78400+ 100+ 1000+10
    caseList = [3]
    nClasses = 10
    nClients = 300
    trainTotalRounds = 200
    lrIni = 0.06
    epochs = 1
    layer = 4
    Groups = 75
    number_per_group = np.int32(nClients / Groups)

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    seed = 1
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



    Group= {}
    i = 0
    J= np.zeros(number_per_group)
    for group in range(Groups):
        Group[group] = np.arange(number_per_group) + i + J[-1]
        i =1
        J = Group[group]
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    param_per_group =np.int32(Total_param/Groups)
    print('param_per_group', param_per_group)
    Segment_length = {}
    i = 0
    for seg in np.arange(Groups - 1):
        Segment_length[seg] = param_per_group + i
        i = Segment_length[seg]
    Segment_length[Groups-1] = layer_0

    #print('Segment_length',Segment_length)

    # Generate the SS selection matrix for MC
    B = np.zeros([Groups,Groups])+1000
    for g in np.arange(Groups -1):
        for r in np.arange(Groups -1-g):
            l = 2*g+r
            B [l%Groups][g] =g
            B [l%Groups][g+r+1]  = g
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

    non_iid = 1
    case = 3
    if case == 1:
        byzant = 0
        attack = 'No-attack'
    elif case == 2:  # main focus of agenda 1
        attack = 'gaussian'  # 1 for non-iid, 0 for iid
        byzant = 1  # 1 for byzantine workers
    elif case == 3:  # main focus of agenda 1
        attack = 'sign_flip'  # 1 for non-iid, 0 for iid
        byzant = 1  # 1 for byzantine workers
    else:
        attack = 'label_flip'  # 1 for non-iid, 0 for iid
        byzant = 1  # 1 for byzantine workers

    case1 = 2
    if case1 == 1:
        Aggregation = 'mean'
    elif case1 == 2:  # main focus of agenda 1
        Aggregation = 'median'  # 1 for non-iid, 0 for iid
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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


    xtr = np.load('mnist-x-trainSorted.npy')
    ytr = np.load('mnist-y-trainSorted.npy')
    trainDataSize = np.size(xtr, 0)
    xte = np.load('mnist-x-test.npy')
    yte = np.load('mnist-y-test.npy')
    testDataSize = np.size(xte, 0)
    print('max xte', np.max(xte))
    print('1 xte type', xte.dtype)
    print('1 yte type', yte.dtype)
    print('1 xte size', np.shape(xte))
    print('1 yte size', np.shape(yte))
    print('1 xtr type', xtr.dtype)
    print('max xtr', np.max(xtr))
    print('1 ytr type', ytr.dtype)
    print('xtr shape', np.shape(xtr))
    print('1 ytr size', np.shape(ytr))
    print('some y samples:', ytr[0:20])
    ytr = np.int64(ytr)
    xtr = np.float32(np.reshape(xtr, (trainDataSize, 1, 28, 28)))
    yte = np.int64(yte)
    xte = np.float32(np.reshape(xte, (testDataSize, 1, 28, 28)))
    xtr = xtr / 255.0
    xtr = (xtr - 0.5) / 0.5
    xte = xte / 255.0
    xte = (xte - 0.5) / 0.5
    xtr = xtr * 1.0
    xtr = torch.from_numpy(xtr)
    ytr = torch.from_numpy(ytr)
    xtr = xtr.to(device)
    ytr = ytr.to(device)
    xte = xte * 1.0
    xte = torch.from_numpy(xte)
    yte = torch.from_numpy(yte)
    xte = xte.to(device)
    yte = yte.to(device)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if non_iid == 0:
        temp1 = np.random.permutation(trainDataSize)
        xtr = xtr[temp1]
        ytr = ytr[temp1]
    print('some y samples:', ytr[0:20])
    print('here',byzantClientsList)
    trainDataSizeFracClients = np.array([1 for iClient3 in range(nClients)])
    trainDataSizeFracClients = trainDataSizeFracClients / np.sum(trainDataSizeFracClients)
    trainDataSizeClients = np.int32(trainDataSizeFracClients * trainDataSize)
    print('trainDataSizeClients:', trainDataSizeClients)
    # time.sleep(3)
    miniBatchSizeFrac = np.array([0.2 for iClient in range(nClients)])
    miniBatchSizeClients = np.int32(miniBatchSizeFrac * trainDataSizeClients)
    clientsEpochs = np.array([1 for iClient in range(nClients)])
    nMiniBatchesClientsPerEpoch = np.int32(trainDataSizeClients / miniBatchSizeClients)

    trainDataIndices = {}
    stIndex = 0
    for iClient in range(nClients):
        trainDataIndices[(iClient, 0)] = stIndex
        trainDataIndices[(iClient, 1)] = (stIndex + trainDataSizeClients[iClient])
        stIndex = (stIndex + trainDataSizeClients[iClient])

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.classifier = nn.Sequential(
                nn.Linear(28 * 28, 100),
                nn.ReLU(inplace=True),
                nn.Linear(100, 10),
            )

        def forward(self, x):
            x = x.view(x.size(0), 28 * 28)
            x = self.classifier(x)
            return x


    net = Net()
    print(Net)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    lr = lrIni
    optimizer = optim.SGD(net.parameters(), lr=lr)
    test_accuracy = []
    modelParamsClientsOld = {}
    modelParamsClientsNew = {}
    modelGradssClients = {}
    vectrorized_gradParams_aggregated_per_group= {}
    vectrorized_gradParams_aggregated={}
    vectrorized_gradParams = {}

    #print(net)
    lcount = 0
    for param in net.parameters():
        for iClient in range(nClients):
            modelParamsClientsOld[(iClient, lcount)] = param.clone().detach()# at the end of previous round
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
            clientDataY = ytr[trainDataIndices[(iClient, 0)]:trainDataIndices[(iClient, 1)]]
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
            # print(nEpochs,'nEpochs')
            nBatches = nMiniBatchesClientsPerEpoch[iClient]
            # print(nBatches,'nBatches')
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

            with torch.no_grad():
                lcount = 0
                for param in net.parameters():
                    # gradParams =Quantizer(gradParams, 2, -1, 1)
                    modelGradssClients[(iClient, lcount)] = param.grad.clone().detach()  # at the end of previous round
                    # modelGradssClients[(iClient, lcount)] =Quantizer( modelGradssClients[(iClient, lcount)], 2 ,-1, 1)
                    modelParamsClientsOld[(iClient, lcount)] = param.data - lr * \
                                                            modelGradssClients[(iClient, lcount)]
                    modelGradssClients[(iClient, lcount)] = modelParamsClientsOld[(iClient, lcount)] - \
                                                            modelParamsClientsNew[(iClient, lcount)]
                    if byzant == 1:
                        if iClient in byzantClientsList:
                            if attack =='gaussian':
                                modelGradssClients[(iClient, lcount)] = 5* torch.randn(np.shape(param))
                            elif attack =='sign_flip':
                                #print(iClient)
                                modelGradssClients[(iClient, lcount)] = -5 * torch.randn(np.shape(param))
                            elif attack =='label_flip':
                                modelGradssClients[(iClient, lcount)] = 30 * modelGradssClients[(iClient, lcount)]
                        else:
                            modelGradssClients[(iClient, lcount)] = modelGradssClients[(iClient, lcount)]
                    else:
                        modelGradssClients[(iClient, lcount)] = modelGradssClients[(iClient, lcount)]
                    lcount = lcount + 1
        for iClient in range(nClients):
            for lcount_loop in np.arange(layer):
                if lcount_loop == 0:
                    for group_index in np.arange(Groups):
                        if iClient in Group[group_index]:
                            vectrorized_gradParams_0 = torch.flatten(modelGradssClients[(iClient, lcount_loop)])
                            i=0
                            for segment_index in np.arange(Groups):
                                vectrorized_gradParams[( lcount_loop,segment_index, iClient, group_index)] = vectrorized_gradParams_0[i:Segment_length[segment_index]]
                                i = Segment_length[segment_index]


                if lcount_loop == 1:
                    for group_index in np.arange(Groups):
                        if iClient in Group[group_index]:
                            segment_index =  Groups-1

                            vectrorized_gradParams[(lcount_loop, segment_index, iClient, group_index)] = modelGradssClients[(iClient, lcount_loop)]


                if lcount_loop == 2:
                    for group_index in np.arange(Groups):
                        if iClient in Group[group_index]:
                            segment_index=  Groups-1
                            vectrorized_gradParams[(lcount_loop, segment_index, iClient, group_index)] = torch.flatten(modelGradssClients[(iClient, lcount_loop)])

                if lcount_loop == 3:
                    for group_index in np.arange(Groups):
                        if iClient in Group[group_index]:
                            segment_index = Groups-1
                            vectrorized_gradParams[(lcount_loop, segment_index, iClient, group_index)] = modelGradssClients[(iClient, lcount_loop)]



        for lcount_loop in np.arange(layer):
            for segment_index in np.arange(Groups):
                for group_index in np.arange(Groups):
                    for iClient in Group[group_index]:
                        if (lcount_loop !=0 and  segment_index ==  Groups-1) :
                            vectrorized_gradParams_aggregated_per_group[(lcount_loop,segment_index, group_index)]=torch.zeros(( vectrorized_gradParams[(lcount_loop, segment_index,np.int(iClient), group_index)].size()))
                            # print(torch.max(vectrorized_gradParams_aggregated_per_group[(lcount_loop,segment_index, group_index)]))
                        elif lcount_loop ==0:
                            vectrorized_gradParams_aggregated_per_group[(lcount_loop,segment_index, group_index)]=torch.zeros(( vectrorized_gradParams[(lcount_loop, segment_index,np.int(iClient), group_index)].size()))






    ############take the average over each group:: output G*G dictionary where each column is the average of one group
        for lcount_loop in np.arange(layer):
            for segment_index in np.arange(Groups):
                for group_index in np.arange(Groups):
                    for iClient in Group[group_index]:
                        if lcount_loop != 0 and segment_index ==  Groups-1 :
                            vectrorized_gradParams_aggregated_per_group[(lcount_loop,segment_index, group_index)]+=1/number_per_group*vectrorized_gradParams[(lcount_loop, segment_index,np.int(iClient), group_index)]


                        elif lcount_loop == 0:
                            vectrorized_gradParams_aggregated_per_group[(lcount_loop, segment_index, group_index)]+=1/number_per_group*vectrorized_gradParams[(lcount_loop, segment_index, np.int(iClient), group_index)]




                            # intialize the dictionary where we put the  gradients average of  groups that have the same index in the SS matrix
        for lcount_loop in np.arange(layer):
            for segment_index in np.arange(Groups):
                for merged_groups_index in np.arange((Groups+1)/2): # number of merged  when groups is odd
                    if lcount_loop != 0:
                        segment_index =  Groups-1
                    vectrorized_gradParams_aggregated[(lcount_loop, segment_index, merged_groups_index)]= torch.zeros(vectrorized_gradParams_aggregated_per_group[  (lcount_loop,segment_index,0)].size())

                ### compute   gradients average of  groups that have the same index in the SS matrix
        for lcount_loop in np.arange(layer):
            for segment_index in np.arange(Groups):
                for merged_groups_index in np.arange((Groups+1)/2): # number of merged  when groups is odd
                    if lcount_loop != 0 and segment_index  == Groups-1 :
                        if np.size(np.array([Collected_groups[(segment_index,merged_groups_index)]]))==2: # since Collected_groups either have two groups or an individual group has *
                            for group_index in  Collected_groups[(segment_index,merged_groups_index)]:
                                vectrorized_gradParams_aggregated[(lcount_loop, segment_index, merged_groups_index)]+= 0.5* vectrorized_gradParams_aggregated_per_group[  (lcount_loop,segment_index,group_index)]



                        else:

                            Bufer = Collected_groups[(segment_index, merged_groups_index)]

                            vectrorized_gradParams_aggregated[(lcount_loop, segment_index, merged_groups_index)]= vectrorized_gradParams_aggregated_per_group[(lcount_loop, segment_index,Bufer)]

                    elif lcount_loop ==0:

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
                if lcount_loop != 0:
                    segment_index = Groups-1
                x = [vectrorized_gradParams_aggregated[(lcount_loop,segment_index, merged_groups_index)].numpy() for merged_groups_index in np.arange((Groups+1)/2)]
                '''if lcount_loop != 0:
                    #print('x', x)
                    print('mena', np.mean(x, axis=0))'''
                if case1 == 1:
                    median_grad[(lcount_loop,segment_index)]=np.mean(x,axis =0 )
                else:
                    median_grad[(lcount_loop,segment_index)]=np.median(x,axis =0 )



        for lcount_loop in np.arange(layer):

            if lcount_loop ==0 :

                modelGradssClients_layer[lcount_loop] = np.concatenate(([median_grad[(lcount_loop, segment_index)] for segment_index in np.arange(Groups)]), axis = 0)


                modelGradssClients_layer[lcount_loop] = np.reshape(modelGradssClients_layer[lcount_loop], (100, -1))
            elif lcount_loop ==1:
                modelGradssClients_layer[lcount_loop]= median_grad[(lcount_loop,  Groups-1)]

            elif lcount_loop == 3:
                modelGradssClients_layer[lcount_loop] = median_grad[(lcount_loop,  Groups-1)]

            else:
                modelGradssClients_layer[lcount_loop] = np.reshape(median_grad[(lcount_loop,  Groups-1)], (10, -1))


        for lcountClient in np.arange(nClients)   :
                modelParamsClientsOld[(iClient, lcount)]=modelParamsClientsNew[(iClient, lcount)]+ modelGradssClients_layer[lcount]




    np.save(
            'Test_accuracy_MNIST_byzant_{0}_Epochs_{1}__non_iid_{2}_case_{3}_attack{4}_Aggregation_{5}ـNumber_Byzantines{6}'.format(byzant, epochs,  non_iid,case,attack,Aggregation,nByzantines), test_accuracy) in np.arange(layer):
            modelGradssClients_layer[lcount] = torch.from_numpy(modelGradssClients_layer[lcount])

parser = argparse.ArgumentParser()
parser.add_argument("--case", type=int, default=1)  # 2:Gaussian, 3:sign flip
parser.add_argument("--case1", type=int, default=1) # 1:mean, 2:median


# parser.add_argument("--connet_pro", type=float,default=0.4)
args = parser.parse_args()

main(args=args)     