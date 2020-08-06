#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import *
import tensorflow as tf
import math


# In[13]:


def load_data():
    data_path="C:\\University Master\\NNs project\\GTSRB_Final_Training_Images_new"
    data_csv_path=os.path.join(data_path,"GT.csv")
    train_path="C:\\University Master\\NNs project\\GTSRB_Final_Training_Images_new\\train.csv"
    test_path="C:\\University Master\\NNs project\\GTSRB_Final_Training_Images_new\\test.csv"
    #print(traing_data_csv_path)
    X_train_orig=[]
    Y_train_orig=[]
    X_valid_orig=[]
    Y_valid_orig=[]
    df = pd.read_csv(data_csv_path)
    df['split'] = np.random.randn(df.shape[0], 1)

    msk = np.random.rand(len(df)) <= 0.9
    #print(msk.size)

    train = df[msk]
    test = df[~msk]
    train_csv = train.to_csv (r'C:\\University Master\\NNs project\\GTSRB_Final_Training_Images_new\\train.csv',index=None, header=True)
    test_csv = test.to_csv (r'C:\\University Master\\NNs project\\GTSRB_Final_Training_Images_new\\test.csv',index=None, header=True)
    #train
    with open(train_path,'r') as csv_file: #Opens the file in read mode
        csv_reader = csv.reader(csv_file) # Making use of reader method for reading the file
        i=0
        for line in csv_reader:#Iterate through the loop to read line by line
            if(i>1):
                img_path=os.path.join(data_path,line[0])
                #print(img_path)
                X_train_orig.append(mpimg.imread(img_path))
                Y_train_orig.append(int(line[1]))
                #print(i)
            i+=1
            
            
    #test
    with open(test_path,'r') as csv_file: #Opens the file in read mode
        csv_reader = csv.reader(csv_file) # Making use of reader method for reading the file
        i=0
        for line in csv_reader:#Iterate through the loop to read line by line
            if(i>1):
                img_path=os.path.join(data_path,line[0])
                #print(img_path)
                X_valid_orig.append(mpimg.imread(img_path))
                Y_valid_orig.append(int(line[1]))
                #print(i)
            i+=1
    X_train_orig=np.array(X_train_orig)
    X_train=X_train_orig/255.
    Y_train_orig=np.array(Y_train_orig)
    #Y_train = np.eye(43)[Y_train_orig.reshape(-1)].T.T
	
    X_valid_orig=np.array(X_valid_orig)
    X_valid= X_valid_orig/255.
    Y_valid_orig=np.array(Y_valid_orig)
	
    #Y_valid = np.eye(43)[Y_valid_orig.reshape(-1)].T.T
    return X_train,Y_train_orig,X_valid,Y_valid_orig


# In[ ]:


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches =floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
	
def get_name_by_index(index_value):
    index_text ={
        0 :"Speed limit 20",
        1 :"Speed limit 30",
        2 :"Speed limit 50",
        3 :"Speed limit 60",
        4 :"Speed limit 70",
        5 :"Speed limit 80",
        6 :"Speed limit not 80 or above",
        7 :"Speed limit 100",
        8 :"Speed limit 120",
        9 :"Overtaking prohibited",
        10 :"Overtaking prohibited for truck",
        11 :"Warning: crossroad with two side roads",
        12 :"You are on a priority road",
        13 :"Give way to all drivers",
        14 :"Stop and give way to all drivers",
        15 :"Entry prohibited",
        16 :"Trucks prohibited",
        17 :"Entry prohibited (road with one-way traffic)",
        18 :"Warning: danger with no specific traffic sign",
        19 :"Warning: A curve to the left",
        20 :"Warning: A curve to the right",
        21 :"Warning: A double curve, first left then right",
        22 :"Warning: bad road surface",
        23 :"Warning: slippery road surface",
        24 :"Warning: road narrowing on the right",
        25 :"Warning: roadworks",
        26 :"Warning: traffic light",
        27 :"Warning: crossing for pedestrians",
        28 :"Warning: children",
        29 :"Warning: cyclists",
        30 :"Warning: snow and sleet",
        31 :"Warning: crossing deer",
        32 :"No parking or waiting",
        33 :"Turning right mandatory",
        34 :"Turning left mandatory",
        35 :"Driving straight ahead mandatory",
        36 :"Driving straight ahead or turning right mandatory",
        37 :"Driving straight ahead or turning left mandatory",
        38 :"Passing right mandatory",
        39 :"Passing left mandatory",
        40 :"Mandatory direction of the roundabout",
        41 :"Overtaking prohibited",
        42 :"Overtaking prohibited for trucks"
        
    }
    return index_text.get(index_value, "not exists")


