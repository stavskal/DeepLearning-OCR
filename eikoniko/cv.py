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


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

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

input_var = T.tensor4('X')
target_var = T.ivector('y')

# Network Definition
network = lasagne.layers.InputLayer((None,1,25,25),input_var)
                                            #filters,size,nonlinearity
network = lasagne.layers.Conv2DLayer(network,64,(3,3),nonlinearity=rectify)
network = lasagne.layers.Conv2DLayer(network,32,(3,3),nonlinearity=rectify)
network = lasagne.layers.Pool2DLayer(network,(3,3),stride=2,mode='max')
network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network,0.3),128,nonlinearity=rectify,
                                    W=lasagne.init.Orthogonal())
network= lasagne.layers.DenseLayer(lasagne.layers.dropout(network,0.3),26,nonlinearity=softmax)

# Loss function
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)

#                  regularization of weights commented out for now
loss = loss.mean()# + 1e-4 * lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)



# create parameter update expressions
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01,
                                            momentum=0.9)

stackedtest=np.dstack(Xtest)
stackedtest=stackedtest.reshape(-1,1,25,25)
ytest = np.array(ytest).astype('int32')

# Converting from list to 3D array  
stacked=np.dstack(imlist)
stacked=stacked.reshape(-1,1,25,25)
Y = np.array(Y).astype('int32')
print(stacked.shape, Y.shape)

# compile training function that updates parameters and returns training loss
train_fn = theano.function([input_var, target_var], loss, updates=updates)


test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
test_loss = test_loss.mean()
# As a bonus, also create an expression for the classification accuracy:
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
    dtype=theano.config.floatX)

# Compile a function performing a training step on a mini-batch (by giving
# the updates dictionary) and returning the corresponding training loss:

# Compile a second function computing the validation loss and accuracy:
val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

for epoch in range(8):
    loss = 0
   # train_batches=0
    for batch in iterate_minibatches(stacked, Y, 200, shuffle=True):
            inputs, targets = batch
            loss += train_fn(inputs, targets)
           # train_batches += 1
           
    print("Epoch %d: Loss %g" % (epoch + 1, loss / len(Y)))

test_err = 0
test_acc = 0
test_batches = 0
for batch in iterate_minibatches(stackedtest, ytest, 200, shuffle=False):
    inputs, targets = batch
    err, acc = val_fn(inputs, targets)
    test_err += err
    test_acc += acc
    test_batches += 1
print("Final results:")
print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
print("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))

print(stacked.shape,Y.shape)

net2.fit(stacked,Y)

ytest=np.array(ytest)

preds=net2.predict(stackedtest)
print classification_report(ytest,preds)