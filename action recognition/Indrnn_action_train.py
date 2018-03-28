from __future__ import print_function
import sys
import argparse
import os

import time

import lasagne
import theano
import numpy as np
import theano.tensor as T

from lasagne.layers import InputLayer,ReshapeLayer,DimshuffleLayer,Gate,BatchNormLayer,DenseLayer,ElemwiseSumLayer

from lasagne.layers import ConcatLayer,NonlinearityLayer,DropoutLayer
from lasagne.layers.normalization import batch_norm
from lasagne.nonlinearities import softmax, rectify,tanh
from lasagne.init import Uniform,Normal,HeNormal

import opts
from Indrnn_action_network import build_indrnn_network as build_rnn_network
import Indrnn_action_network

sys.setrecursionlimit(10000)

parser = argparse.ArgumentParser(description='lstm action')
opts.train_opts(parser)
args = parser.parse_args()
print (args)  


batch_size = args.batch_size
seq_len=args.seq_len
outputclass=60
indim=50#150
lr=args.lr
opti=lasagne.updates.adam
U_bound=Indrnn_action_network.U_bound

X_sym = T.tensor4('inputs')#,dtype=theano.config.floatX)
y_sym = T.ivector('label')#,dtype=theano.config.floatX)   
learn_net=build_rnn_network(X_sym)
prediction = lasagne.layers.get_output(learn_net['out'], X_sym,deterministic=False)
loss = lasagne.objectives.categorical_crossentropy(prediction, y_sym).mean()
if args.use_weightdecay_nohiddenW:
  params = lasagne.layers.get_all_params(learn_net['out'], regularizable=True)
  for para in params:
    if para.name!='hidden_to_hidden.W':
      loss += args.decayrate *lasagne.regularization.apply_penalty(para, lasagne.regularization.l2)#*T.clip(T.abs_(para)-1,0,100))  
acc=T.mean(lasagne.objectives.categorical_accuracy(prediction, y_sym, top_k=1),dtype=theano.config.floatX)

params = lasagne.layers.get_all_params(learn_net['out'], trainable=True)
learning_ratetrain = T.scalar(name='learning_ratetrain',dtype=theano.config.floatX)
grads = theano.grad(loss, params)
updates = opti( grads, params, learning_rate=learning_ratetrain)
print('Compiling')
train_fn = theano.function([X_sym, y_sym,learning_ratetrain], [loss,acc], updates=updates)

test_prediction = lasagne.layers.get_output(learn_net['out'], X_sym,deterministic=True,batch_norm_use_averages=False)
test_acc=T.mean(lasagne.objectives.categorical_accuracy(test_prediction, y_sym, top_k=1),dtype=theano.config.floatX)
test_fn = theano.function([X_sym, y_sym], [test_acc,test_prediction]) 

bn_test_prediction = lasagne.layers.get_output(learn_net['out'], X_sym,deterministic=True)#,batch_norm_use_averages=True
bn_test_acc=T.mean(lasagne.objectives.categorical_accuracy(bn_test_prediction, y_sym, top_k=1),dtype=theano.config.floatX)
bn_test_fn = theano.function([X_sym, y_sym], [bn_test_acc,bn_test_prediction])       
  

learning_rate=np.float32(lr)
if args.test_CV:
  train_datasets='train_CV_ntus'
  test_dataset='test_CV_ntus'
else:
  train_datasets='train_ntus'
  test_dataset='test_ntus'   
from data_reader_numpy_witheval import DataHandler_train,DataHandler_eval  
from data_reader_numpy_test import DataHandler as testDataHandler
dh_train = DataHandler_train(batch_size,seq_len, args.rotation_aug)
dh_eval = DataHandler_eval(batch_size,seq_len)
dh_test= testDataHandler(batch_size,seq_len)
num_train_batches=int(np.ceil(dh_train.GetDatasetSize()/(batch_size+0.0)))
num_eval_batches=int(np.ceil(dh_eval.GetDatasetSize()/(batch_size+0.0)))
num_test_batches=int(np.ceil(dh_test.GetDatasetSize()/(batch_size+0.0)))
labelname='test_ntus_label.npy'
if args.test_CV:
  labelname='test_CV_ntus_label.npy'
testlabels=np.load(labelname)

aveloss=0
aveacc=0
lastacc=0
dispFreq=20
testnos=20
stepcount=0   
patience=0
patienceThre=10 
while True:
  x, y = dh_train.GetBatch()
  loss,acc=train_fn(x, y,learning_rate)
  stepcount+=1
  aveloss+=loss
  aveacc+=acc
  
  if args.constrain_U:
    for para in params:
      if para.name=='hidden_to_hidden.W':
        para.set_value(np.clip(para.get_value(),-U_bound,U_bound))
  
  if np.isnan(loss):
    print ('NaN detected in cost')
    assert(2==3)
  if np.isinf(loss):
    print ('INF detected in cost')
    assert(2==3) 

  if np.mod(stepcount, dispFreq) == 0:
    aveloss=aveloss/dispFreq
    aveacc=aveacc/dispFreq
    print("lr",learning_rate,"trainingerror",aveloss,"aveacc",aveacc)
    aveloss=0
    aveacc=0
       
  if np.mod(stepcount, num_train_batches)==0:   
    stepcount=0   
    aveacc=0
    eval_batches=num_eval_batches*args.eval_fold
    for testi in range(eval_batches):
      x, y = dh_eval.GetBatch()
      test_acc_top1,_=test_fn(x, y)      
      aveacc+=test_acc_top1   
    bn_aveacc=0
    for testi in range(eval_batches):
      x, y = dh_eval.GetBatch()
      bn_test_acc_top1,_=bn_test_fn(x, y)      
      bn_aveacc+=bn_test_acc_top1         
      
    print ('evalacc,bn_evalacc', aveacc/eval_batches, bn_aveacc/eval_batches) 
    epocacc=bn_aveacc/eval_batches
    aveacc=0
    
    if (epocacc >lastacc):# and itericount>=0.8*rateschedulecount
      best_para=lasagne.layers.get_all_param_values(learn_net['out'])  
      lastacc=epocacc
      patience=0
    elif patience>patienceThre:
      #learning_rate=np.float32(learning_rate*0.2)
      print ('learning rate',learning_rate)
      lasagne.layers.set_all_param_values(learn_net['out'], best_para)
      patience=0
      learning_rate=np.float32(learning_rate*0.1)    
      if learning_rate<args.end_rate:
        break
    else:
      patience+=1        
    


total_testdata=dh_test.GetDatasetSize()  
total_ave_acc=np.zeros((total_testdata,outputclass))
test_no=10
aveacc=0
for testi in range(num_test_batches*test_no):
  x, y,index = dh_test.GetBatch()
  test_acc_top1,test_prediction=test_fn(x, y)      
  aveacc+=test_acc_top1
  total_ave_acc[index]+=test_prediction
total_ave_acc/=float(test_no)
top = np.argmax(total_ave_acc, axis=-1)
eval_acc=np.mean(np.equal(top, testlabels))
print ('testacc', aveacc/(test_no*num_test_batches), eval_acc)

total_ave_acc=np.zeros((total_testdata,outputclass))
aveacc=0
for testi in range(num_test_batches*test_no):
  x, y,index = dh_test.GetBatch()
  test_acc_top1,test_prediction=bn_test_fn(x, y)      
  aveacc+=test_acc_top1
  total_ave_acc[index]+=test_prediction
total_ave_acc/=float(test_no)
top = np.argmax(total_ave_acc, axis=-1)
eval_acc=np.mean(np.equal(top, testlabels))
print ('bn_testacc', aveacc/(test_no*num_test_batches), eval_acc)

save_name='action_indrnn'  
np.savez(save_name, *lasagne.layers.get_all_param_values(learn_net['out']))
