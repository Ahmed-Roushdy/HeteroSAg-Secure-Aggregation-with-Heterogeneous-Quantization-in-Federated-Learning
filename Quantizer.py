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
import bisect
import random


'''def Quantizer(X, K, min_input, max_input ):
    global X_quantized
    X = X.to('cpu')
    Delta = (max_input - min_input)/np.subtract(K, 1)
    L_ = np.linspace(start=0, stop=np.subtract(K, 1), num=K)
    L__ = L_ * Delta
    Rounded = np.add(  L__,min_input)
    Intervals =np.array([])
    X_final = torch.tensor([]).to(device)
    for i in range(np.subtract(K, 1)):
        Intervals = np.concatenate([Intervals,[Rounded[i], Rounded[i+1]] ])
    Intervals = Intervals.reshape(np.subtract(K, 1),2)
    if X.ndim == 1:
        for element in X:
            for interv in Intervals:
                if (interv[0] <= element < interv[1]) or (interv[0] < element <= interv[1]):
                    B_1 =  interv[0]
                    B_2 = interv[1]
                    prob = np.subtract(element, B_1) / (np.subtract(B_2, B_1))
                    random_ = random.uniform(0, 1)                    #print(random_)
                    if random_ <= prob:
                        ZZ = np.float32(np.array([B_2]))
                        X_quantized = torch.from_numpy(ZZ).to(device)
                    else:
                        ZZ = np.float32(np.array([B_1]))
                        X_quantized = torch.from_numpy(ZZ).to(device)
            #print(X_final.type(), X_quantized.type())
            X_final = torch.cat((X_final, X_quantized))
            #X_final = np.append(X_final, X_quantized)
        X_final = X_final

    else:
        for row in X:
            for element in row:
                # print(element)
                for interv in Intervals:
                    if (interv[0] <= element < interv[1]) or (interv[0] < element <= interv[1]):
                        B_1 = interv[0]
                        B_2 = interv[1]
                        prob = np.subtract(element, B_1) / (np.subtract(B_2, B_1))
                        random_ = random.uniform(0, 1)  # print(random_)
                        if random_ <= prob:
                            ZZ = np.float32(np.array([B_2]))
                            X_quantized = torch.from_numpy(ZZ).to(device)
                        else:
                            ZZ =np.float32( np.array([B_1]))
                            X_quantized = torch.from_numpy(ZZ).to(device)

                        # print(type(X_final), type(torch.from_numpy(X_quantized)))
                #print(X_final.type(), X_quantized.type())
                X_final = torch.cat((X_final, X_quantized))
        X_final =  torch.reshape(X_final, list(X.size()))
    return X_final '''


device = 'cuda:5' if torch.cuda.is_available() else 'cpu'
def Quantizer(X, K, min_input, max_input ):
    if K=='no':
        return X
    else:
        global X_quantized
        min_input = min_input - 10**-6  # avoid any error with 
        max_input = max_input + 10**-6  # avoid any error with 

        X = X.to('cpu')
        Delta = (max_input - min_input)/np.subtract(K, 1)
        L_ = np.linspace(start=0, stop=np.subtract(K, 1), num=K)
        L__ = L_ * Delta
        Rounded = np.add(  L__,min_input)
        Intervals =np.array([])
        X_final = torch.tensor([]).to(device)
        for i in range(np.subtract(K, 1)):
            Intervals = np.concatenate([Intervals,[Rounded[i], Rounded[i+1]] ])
            #print(Intervals)
        #Intervals = Intervals.reshape(np.subtract(K, 1),2)
        for element in X:
            #print(element)
            #print(X.ndim == 1)
            #print(X.size())
            element = element.item()
            a = bisect.bisect_left(Intervals, element)
            #print(a)
            B_1 =  Intervals[a-1]
            B_2 = Intervals[a]
            prob = np.subtract(element, B_1) / (np.subtract(B_2, B_1))
            random_ = random.uniform(0, 1)                    #print(random_)
            if random_ <= prob:
                ZZ = np.float32(np.array([B_2]))
                X_quantized = torch.from_numpy(ZZ).to(device)
            else:
                ZZ = np.float32(np.array([B_1]))
                X_quantized = torch.from_numpy(ZZ).to(device)
            #print(X_final.type(), X_quantized.type())
            X_final = torch.cat((X_final, X_quantized))
            #X_final = np.append(X_final, X_quantized)
        return X_final







device = 'cuda:7' if torch.cuda.is_available() else 'cpu'
def Quantizer_1(X, K, min_input, max_input ):
    if K=='no':
        return X
    else:
        global X_quantized
        #print('max_x', torch.max(X))
        #print('before min_input', min_input)
        #print('before max_input', max_input)

        min_input = min_input +np.sign(min_input)* 10**-6  # avoid any error with 
        max_input = max_input + np.sign(max_input)*10**-6
        #print('after min_input', min_input)
        #print('after max_input', max_input)

        X = X.cpu()
        X = X.numpy()
        Delta = (max_input - min_input)/np.subtract(K, 1)

        L_ = np.linspace(start=0, stop=np.subtract(K, 1), num=K)
        L__ = L_ * Delta
        Rounded = np.add(  L__,min_input)
        #print('Rounded',Rounded)

        
        a = np.searchsorted(Rounded,X)
        #print('a',np.max(a))
        #print('a',np.argmax(a))
        test = np.argmax(a)
        #print('X', X[test])


        B_R  = Rounded[a]
        B_L = Rounded[a-1]
        #print(Rounded[a])
        #print(Rounded[a-1])

        prob = np.subtract(X, B_L) / (np.subtract(B_R, B_L))
        #print('prob',np.isnan(prob).any())
        random_ = np.random.uniform(0, 1, X.shape[0])                   
        #print(random_)
        d =  random_  - prob

        #print('d', np.isnan(d).any())
        # time.sleep(.07)

        B_R_location = np.where(d < 0)
        B_L_location = np.where(d >= 0)
        B_R [B_L_location] = 0

        B_L[B_R_location]  = 0 

        x_quant = B_L+B_R
        #print('x_quant',np.isnan(x_quant).any())
        #print('x_quant',np.max(x_quant))
        x_quant = torch.from_numpy(x_quant)
        x_quant = x_quant.float()

        return x_quant


        