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
from Quantizer import *
import argparse

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def main(args):
    nClients = args.nClients
    Quantization_type = args.Quantization_type # Homo, no Hetro
    lrIni = args.lr
    K_0 = args.K_0
    K_1 = args.K_1
    K_2 = args.K_2
    K_3 = args.K_3
    K_4 = args.K_4
   

    device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
    Epochs_num = 5
    batch_size = 0.1
    lrIni = 0.03
    nClasses = 10
    trainTotalRounds = 200
    Groups = 5
    number_per_group = nClients / Groups
    Group_0 = np.arange(number_per_group)
    Group_1 = np.arange(number_per_group) +Group_0[ -1]+1
    Group_2 = np.arange(number_per_group)+Group_1[ -1]+1
    Group_3 = np.arange(number_per_group)+Group_2[ -1]+1
    Group_4 = np.arange(number_per_group)+Group_3[ -1]+1

    # Segment_0_length = total number of parameter over G. The following will be used as a pointer
    Segment_0_length= 15902
    Segment_1_length= 15902+Segment_0_length
    Segment_2_length= 15902+Segment_1_length
    Segment_3_length= 15902+Segment_2_length
    Segment_4_length= 15902+Segment_3_length
    
    if Quantization_type == 'Homo':
        Quantizers = K_0, K_1, K_2, K_3, K_4
    elif Quantization_type == 'no':
        K_0, K_1, K_2, K_3, K_4 = 'no', 'no', 'no', 'no', 'no'
        Quantizers = 'no'
    elif Quantization_type == 'Hetro':
        Quantizers = K_0, K_1, K_2, K_3, K_4

    else:
        print('wrong choice, please choice between (Homo,no, Hetro ')

    case = 2
    if case == 1:
        non_iid = 0 # 1 for non-iid, 0 for iid  # 1 for byzantine workers
    if case == 2: # main focus of agenda 1
        non_iid = 1 # 1 for non-iid, 0 for iid
    Quantizers = 'Case B 2, 12, 14, 16, 18'


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    seed = 1
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    xtr = np.load('mnist-x-trainSorted.npy')
    ytr = np.load('mnist-y-trainSorted.npy')
    trainDataSize = np.size(xtr,0)
    xte = np.load('mnist-x-test.npy')
    yte = np.load('mnist-y-test.npy')
    testDataSize = np.size(xte,0)
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
    xtr = np.float32(np.reshape(xtr,(trainDataSize,1,28,28)))
    yte = np.int64(yte)
    xte = np.float32(np.reshape(xte,(testDataSize,1,28,28)))
    xtr = xtr/255.0
    xtr = (xtr-0.5)/0.5
    xte = xte/255.0
    xte = (xte-0.5)/0.5
    xtr = xtr*1.0
    xtr = torch.from_numpy(xtr)
    ytr = torch.from_numpy(ytr)
    xtr = xtr.to(device)
    ytr = ytr.to(device)
    xte = xte*1.0
    xte = torch.from_numpy(xte)
    yte = torch.from_numpy(yte)
    xte = xte.to(device)
    yte = yte.to(device)
    if non_iid == 0:
        temp1 = np.random.permutation(trainDataSize)
        xtr = xtr[temp1]
        ytr = ytr[temp1]
    print('some y samples:', ytr[0:20])
    nByzantines = 0 # agenda 1 for no byzantines
    byzantClientsList =np.arange(0)



    trainDataSizeFracClients = np.array([1 for iClient3 in range(nClients)])
    trainDataSizeFracClients = trainDataSizeFracClients/np.sum(trainDataSizeFracClients)
    trainDataSizeClients = np.int32(trainDataSizeFracClients*trainDataSize)
    print('trainDataSizeClients:', trainDataSizeClients)
    # time.sleep(3)
    miniBatchSizeFrac = np.array([batch_size for iClient in range(nClients)])
    miniBatchSizeClients = np.int32(miniBatchSizeFrac*trainDataSizeClients)
    clientsEpochs = np.array([Epochs_num for iClient in range(nClients)])
    nMiniBatchesClientsPerEpoch = np.int32(trainDataSizeClients / miniBatchSizeClients)

    trainDataIndices = {}
    stIndex = 0
    for iClient in range(nClients):
        trainDataIndices[(iClient,0)] = stIndex
        trainDataIndices[(iClient,1)] = (stIndex + trainDataSizeClients[iClient])
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
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    lr = lrIni
    optimizer = optim.SGD(net.parameters(), lr = lr)
    test_accuracy = []
    modelParamsClientsOld = {}
    modelParamsClientsNew = {}
    modelGradssClients = {}
    print(net)
    lcount = 0
    for param in net.parameters():
        for iClient in range(nClients):
            modelParamsClientsOld[(iClient,lcount)] = param.clone().detach() # at the end of previous round
            modelParamsClientsNew[(iClient,lcount)] = param.clone().detach()
        lcount = lcount+1

    for iRound in range(trainTotalRounds):
        lr = lrIni
        for iClient in range(nClients):
            lcount = 0
            for param in net.parameters():
                modelParamsClientsNew[(iClient, lcount)].fill_(0)
                for client4 in np.arange(nClients):
                    modelParamsClientsNew[(iClient, lcount)] += 1 / nClients * modelParamsClientsOld[(client4, lcount)]
                lcount = lcount + 1

        # np.save(
        #     'modelParamsClientsNew_Quatization_type_{0}_Batch_size_{1}_Quantizers_{2}_Epochs_{3}_non_iid_{4}_learning_rate_{5}'.format(
        #         Quatization_type,batch_size ,Quantizers,Epochs_num, non_iid, lrIni), modelParamsClientsNew)
        with torch.no_grad():
            lcount = 0
            for param in net.parameters():
                param.data = modelParamsClientsNew[(1, lcount)].clone().detach()
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

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        Max_ = []
        Min_ = []

        for iClient in range(nClients):
            with torch.no_grad():
                lcount = 0
                for param in net.parameters():
                    param.data = modelParamsClientsNew[(1,lcount)].clone().detach()
                    lcount = lcount+1
            clientDataX = xtr[trainDataIndices[(iClient, 0)]:trainDataIndices[(iClient, 1)]]
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
    ########################### Store all clinets gradient and get [r1,r2] hte range of the gradinet along all the clients#####

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

                    a = torch.max(modelGradssClients[(iClient, lcount)]).to('cpu')
                    b= torch.min(modelGradssClients [(iClient, lcount)]).to('cpu')
                    Max_ = np.append(Max_, a)
                    Min_ = np.append(Min_, b)
                    lcount = lcount + 1

        max = np.max(Max_)
        min = np.min(Min_)

        vectrorized_gradParams_0= torch.tensor([])
        ## Loop over the stored set of gradient to do quantization and model update##########
        for iClient in range(nClients):
            for lcount_loop  in np.arange(4):
                #print(lcount_loop)
                if lcount_loop == 0:
                    #print(modelGradssClients [(iClient, lcount_loop)].shape)
                    vectrorized_gradParams_0= torch.reshape(modelGradssClients [(iClient, lcount_loop)],(1,78400))
                    gradParams_segment_0 = vectrorized_gradParams_0[0][0:Segment_0_length]
                    gradParams_segment_1 = vectrorized_gradParams_0[0][Segment_0_length:Segment_1_length]
                    gradParams_segment_2 = vectrorized_gradParams_0[0][Segment_1_length:Segment_2_length]
                    gradParams_segment_3 = vectrorized_gradParams_0[0][Segment_2_length:Segment_3_length]
                    gradParams_segment_4 = vectrorized_gradParams_0[0][Segment_3_length:78400]


                    if iClient in Group_0:
                        gradParams_segment_0_quantized = Quantizer(gradParams_segment_0, K_0, min, max)
                        gradParams_segment_1_quantized = Quantizer(gradParams_segment_1, K_0, min, max)
                        gradParams_segment_2_quantized = Quantizer(gradParams_segment_2, K_0, min, max)
                        gradParams_segment_3_quantized = Quantizer(gradParams_segment_3, K_0, min, max)
                        gradParams_segment_4_quantized = Quantizer(gradParams_segment_4, K_0, min, max)
                        #print(list(gradParams_segment_0_quantized.size()), list(gradParams_segment_1_quantized.size()), list(gradParams_segment_2_quantized.size()),list(gradParams_segment_3_quantized.size()),list(gradParams_segment_4_quantized.size()))

                        vectrorized_gradParams_0= torch.cat((gradParams_segment_0_quantized, gradParams_segment_1_quantized, gradParams_segment_2_quantized,gradParams_segment_3_quantized,gradParams_segment_4_quantized))
                        modelGradssClients [(iClient, lcount_loop)]= torch.reshape(vectrorized_gradParams_0, (100, -1))
                    elif iClient in Group_1:
                        gradParams_segment_0_quantized = Quantizer(gradParams_segment_0, K_0, min, max)
                        gradParams_segment_1_quantized = Quantizer(gradParams_segment_1, K_0, min, max)
                        gradParams_segment_2_quantized = Quantizer(gradParams_segment_2, K_1, min, max)
                        gradParams_segment_3_quantized = Quantizer(gradParams_segment_3, K_1, min, max)
                        gradParams_segment_4_quantized = Quantizer(gradParams_segment_4, K_1, min, max)
                        vectrorized_gradParams_0 = torch.cat((gradParams_segment_0_quantized,
                                                            gradParams_segment_1_quantized,
                                                            gradParams_segment_2_quantized,
                                                            gradParams_segment_3_quantized,
                                                            gradParams_segment_4_quantized))
                        modelGradssClients[(iClient, lcount_loop)] = torch.reshape(vectrorized_gradParams_0, (100, -1))

                    elif iClient in Group_2:
                        gradParams_segment_0_quantized = Quantizer(gradParams_segment_0, K_2, min, max)
                        gradParams_segment_1_quantized = Quantizer(gradParams_segment_1, K_0, min, max)
                        gradParams_segment_2_quantized = Quantizer(gradParams_segment_2, K_1, min, max)
                        gradParams_segment_3_quantized = Quantizer(gradParams_segment_3, K_2, min, max)
                        gradParams_segment_4_quantized = Quantizer(gradParams_segment_4, K_2, min, max)
                        vectrorized_gradParams_0 = torch.cat((gradParams_segment_0_quantized,
                                                            gradParams_segment_1_quantized,
                                                            gradParams_segment_2_quantized,
                                                            gradParams_segment_3_quantized,
                                                            gradParams_segment_4_quantized))
                        modelGradssClients[(iClient, lcount_loop)] = torch.reshape(vectrorized_gradParams_0, (100, -1))
                    elif iClient in Group_3:
                        gradParams_segment_0_quantized = Quantizer(gradParams_segment_0, K_3, min, max)
                        gradParams_segment_1_quantized = Quantizer(gradParams_segment_1, K_3, min, max)
                        gradParams_segment_2_quantized = Quantizer(gradParams_segment_2, K_0, min, max)
                        gradParams_segment_3_quantized = Quantizer(gradParams_segment_3, K_1, min, max)
                        gradParams_segment_4_quantized = Quantizer(gradParams_segment_4, K_2, min, max)
                        vectrorized_gradParams_0 = torch.cat(
                            (gradParams_segment_0_quantized, gradParams_segment_1_quantized, gradParams_segment_2_quantized,
                            gradParams_segment_3_quantized, gradParams_segment_4_quantized))
                        modelGradssClients[(iClient, lcount_loop)] = torch.reshape(vectrorized_gradParams_0, (100, -1))
                    else:
                        gradParams_segment_0_quantized = Quantizer(gradParams_segment_0, K_2, min, max)
                        gradParams_segment_1_quantized = Quantizer(gradParams_segment_1, K_3, min, max)
                        gradParams_segment_2_quantized = Quantizer(gradParams_segment_2, K_4, min, max)
                        gradParams_segment_3_quantized = Quantizer(gradParams_segment_3, K_0, min, max)
                        gradParams_segment_4_quantized = Quantizer(gradParams_segment_4, K_1, min, max)
                        vectrorized_gradParams_0 = torch.cat(
                            (gradParams_segment_0_quantized, gradParams_segment_1_quantized, gradParams_segment_2_quantized,
                            gradParams_segment_3_quantized, gradParams_segment_4_quantized))
                        modelGradssClients[(iClient, lcount_loop)] = torch.reshape(vectrorized_gradParams_0, (100, -1))
                elif lcount_loop == 1 or lcount_loop == 3:
                    if iClient in Group_0:
                        modelGradssClients[(iClient, lcount_loop)] = Quantizer(modelGradssClients[(iClient, lcount_loop)], K_0, min, max)
                    elif iClient in Group_1:
                        modelGradssClients[(iClient, lcount_loop)] = Quantizer(modelGradssClients[(iClient, lcount_loop)], K_1, min, max)
                    elif iClient in Group_2:
                        modelGradssClients[(iClient, lcount_loop)] = Quantizer(modelGradssClients[(iClient, lcount_loop)], K_2, min, max)
                    elif iClient in Group_3:
                        modelGradssClients[(iClient, lcount_loop)] = Quantizer(modelGradssClients[(iClient, lcount_loop)], K_2, min, max)
                    else:
                        modelGradssClients[(iClient, lcount_loop)] = Quantizer(modelGradssClients[(iClient, lcount_loop)], K_1, min, max)

                else:
                    modelGradssClients[(iClient, lcount_loop)] = torch.reshape(modelGradssClients[(iClient, lcount_loop)],
                                                                                (1, 1000))
                    modelGradssClients[(iClient, lcount_loop)] = modelGradssClients[(iClient, lcount_loop)][0][0:1000]

                    if iClient in Group_0:
                        modelGradssClients[(iClient, lcount_loop)] = Quantizer(modelGradssClients[(iClient, lcount_loop)], K_0, min, max)
                        modelGradssClients[(iClient, lcount_loop)] = torch.reshape(modelGradssClients[(iClient, lcount_loop)],
                                                                                (10, -1))


                    elif iClient in Group_1:
                        modelGradssClients[(iClient, lcount_loop)] = Quantizer(modelGradssClients[(iClient, lcount_loop)], K_1, min, max)
                        modelGradssClients[(iClient, lcount_loop)] =torch.reshape(modelGradssClients[(iClient, lcount_loop)],
                                                                                (10, -1))


                    elif iClient in Group_2:
                        modelGradssClients[(iClient, lcount_loop)] = Quantizer(modelGradssClients[(iClient, lcount_loop)], K_2, min, max)
                        modelGradssClients[(iClient, lcount_loop)] = torch.reshape(modelGradssClients[(iClient, lcount_loop)],
                                                                                (10, -1))


                    elif iClient in Group_3:
                        modelGradssClients[(iClient, lcount_loop)] = Quantizer(modelGradssClients[(iClient, lcount_loop)], K_2, min, max)
                        modelGradssClients[(iClient, lcount_loop)] = torch.reshape(modelGradssClients[(iClient, lcount_loop)],
                                                                                (10, -1))

                    else:
                        modelGradssClients[(iClient, lcount_loop)] = Quantizer(modelGradssClients[(iClient, lcount_loop)], K_1, min, max)
                        modelGradssClients[(iClient, lcount_loop)] = torch.reshape(modelGradssClients[(iClient, lcount_loop)],
                                                                                (10, -1))


                modelParamsClientsOld[(iClient, lcount_loop)] = modelGradssClients[(iClient, lcount_loop)] + modelParamsClientsNew[(1, lcount_loop)]
    np.save(
        'Test_accuracy_Quatization_type_{0}_Batch_size_{1}_Quantizers_{2}_Epochs_{3}_non_iid_{4}_learning_rate_{5}_nClients{6}'.format(
            Quatization_type, batch_size, Quantizers, Epochs_num, non_iid, lrIni,nClients), test_accuracy)


parser = argparse.ArgumentParser()
parser.add_argument("--nClients", type=int, default=25)
parser.add_argument("--Quantization_type",type=str)
parser.add_argument("--K_0", type=int)
parser.add_argument("--K_1", type=int)
parser.add_argument("--K_2", type=int)
parser.add_argument("--K_3", type=int)
parser.add_argument("--K_4", type=int)





# parser.add_argument("--connet_pro", type=float,default=0.4)
args = parser.parse_args()

main(args=args)
