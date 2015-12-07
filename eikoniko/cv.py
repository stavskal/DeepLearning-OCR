import json,csv,sys,os,psycopg2
import numpy as np
from collections import Counter 
#from processingFunctions import *
import matplotlib.pyplot as pyp
import time

import theano
import theano.tensor as T
from nolearn.lasagne import NeuralNet, TrainSplit, objective
from nolearn.lasagne.visualize import plot_loss
import lasagne
#from lasagne.layers import InputLayer
#from lasagne.layers import DenseLayer
from lasagne import layers
from lasagne.layers import get_all_params
from lasagne.nonlinearities import softmax,sigmoid,rectify
from lasagne.objectives import categorical_crossentropy
import seaborn as sns
from sklearn import preprocessing, linear_model
from sklearn.cross_validation import cross_val_score, KFold
import pandas as pd
from sklearn.metrics import classification_report
from scipy.signal import medfilt
from skimage import io,color
from scipy import misc




def regularization_objective(layers, lambda1=0., lambda2=0., *args, **kwargs):
    # default loss
    losses = objective(layers, *args, **kwargs)
    # get the layers' weights, but only those that should be regularized
    # (i.e. not the biases)
    weights = get_all_params(layers[-1], regularizable=True)
    # sum of absolute weights for L1
    sum_abs_weights = sum([abs(w).sum() for w in weights])
    # sum of squared weights for L2
    sum_squared_weights = sum([(w ** 2).sum() for w in weights])
    # add weights to regular loss
    losses += lambda1 * sum_abs_weights + lambda2 * sum_squared_weights
    return losses



net2 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('hidden3', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
        input_shape = (None,1,25,25),
        conv1_num_filters=10, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
        conv2_num_filters=10, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
        hidden3_num_units=300,
        output_num_units=26,
        output_nonlinearity=softmax,
        update_learning_rate=0.01,
        
        use_label_encoder=True,
  	    update_momentum=0.9,
  	    regression=False,
   	    max_epochs=100,
 	    verbose=1,
        )

os.chdir('/media/tabrianos/90BEA0A3BEA08378/dataset/TRAIN')
directory = os.path.dirname(os.path.abspath(__file__)) 
#print(os.listdir(directory))
Y=[]
imlist=[]
Xtest=[]
ytest =[]
i=0

for filename in os.listdir(directory):
	# extracting labels from filename
    label = filename[-8:-7]
    if ord(label)>=65 and ord(label)<=90: 
       # labelVec=np.zeros(26)
        cl = ord(label)-65
       # labelVec[cl] = cl
        
        tempim = np.array(color.rgb2gray(io.imread(filename)))
        #print(np.unique(tempim))
        # keeping only the sub-matrix that contains the non-white pixels
        # then resizing to feed into the DNN
        index=np.where(tempim==0)
        new_im = tempim[min(index[0]):max(index[0])+1, min(index[1]):max(index[1])+1]
        new_im = misc.imresize(new_im,[25,25])
        if i>=3000 and i <=4000:
            Xtest.append(new_im)
            ytest.append(cl)
        else:
            imlist.append(new_im)
            Y.append(cl)
        if i>4000:
            break
    i += 1
    print(i)
    


# Converting from list to 3D array	
stacked=np.dstack(imlist)
stacked=stacked.reshape(-1,1,25,25)
Y = np.array(Y)
print(stacked.shape,Y.shape)

net2.fit(stacked,Y)

ytest=np.array(ytest)
stackedtest=np.dstack(Xtest)
stackedtest=stackedtest.reshape(-1,1,25,25)
preds=net2.predict(stackedtest)
print classification_report(ytest,preds)