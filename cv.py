import json,csv,sys,os,psycopg2
import numpy as np
from collections import Counter 
#from processingFunctions import *
import matplotlib.pyplot as pyp
import time
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.linear_model import LogisticRegression ,LinearRegression
import theano
import theano.tensor as T
from nolearn.lasagne import NeuralNet, TrainSplit
from nolearn.lasagne.visualize import plot_loss
import lasagne
#from lasagne.layers import InputLayer
#from lasagne.layers import DenseLayer
from lasagne import layers
import seaborn as sns
from sklearn import preprocessing, linear_model
from sklearn.cross_validation import cross_val_score, KFold
import pandas as pd
from scipy.signal import medfilt
from skimage import io,color
from scipy import misc

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
        output_nonlinearity=None,
        update_learning_rate=0.01,
  	    update_momentum=0.9,
  	    regression=True,
   	    max_epochs=100,
 	    verbose=1,
        )

os.chdir('/media/tabrianos/90BEA0A3BEA08378/TRAIN')
directory = os.path.dirname(os.path.abspath(__file__)) 
print(os.listdir(directory))
Y=[]
imlist=[]
#X= np.empty((25,25),dtype='float32')
i=0
for filename in os.listdir(directory):
	label = filename[-8:-7]
	#print(label)
	if ord(label)>=65 and ord(label)<=90: 
		Y.append( filename[-8:-7] )
		tempim = np.array(color.rgb2gray(io.imread(filename)) )

		index=np.where(tempim==0)
		new_im = tempim[min(index[0]):max(index[0])+1, min(index[1]):max(index[1])+1]
		new_im = misc.imresize(new_im,[25,25])
		imlist.append(new_im)
	
stacked=np.dstack(imlist)
print(stacked.shape)
