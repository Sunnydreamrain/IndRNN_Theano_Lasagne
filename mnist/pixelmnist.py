from __future__ import print_function
import sys
import argparse
import os

import time
from collections import OrderedDict
import lasagne
import theano
import numpy as np
import theano.tensor as T

from lasagne.layers import InputLayer,ReshapeLayer,DimshuffleLayer,Gate,DenseLayer,ElemwiseSumLayer
from lasagne.layers import ConcatLayer,NonlinearityLayer,DropoutLayer,BatchNormLayer
from lasagne.nonlinearities import softmax, rectify,tanh,leaky_rectify
from lasagne.init import Uniform, Normal,HeNormal

from lasagne.layers import RecurrentLayer,LSTMLayer
from IndRNN_onlyrecurrent import IndRNNLayer_onlyrecurrent as indrnn_onlyrecurrent

parser = argparse.ArgumentParser(description='IndRNN solving the pixel MNIST problem')
parser.add_argument('--model', type=str, default='indrnn', help='models')
parser.add_argument('--num_layers', type=int, default=6)
parser.add_argument('--hidden_units', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=np.float32, default=2e-4, help='lr')
parser.add_argument('--ini', type=np.float32, default=0.001, help='ini')
parser.add_argument('--gradclipvalue', type=np.float32, default=10)
parser.add_argument('--use_permute', action='store_true', default=False)
parser.add_argument('--use_weightdecay_nohiddenW', action='store_true', default=False)
parser.add_argument('--decayrate', type=np.float32, default=5e-4,help='lr')
parser.add_argument('--ini_b', type=np.float32, default=0.0)
parser.add_argument('--MAG', type=int, default=2)


parser.add_argument('--use_bn_afterrnn', action='store_true', default=False)
args = parser.parse_args()
print (args)


batch_size = args.batch_size
hidden_units=args.hidden_units
outputclass=10

from Data_gen import DataHandler,testDataHandler
if args.use_permute:
  from Data_gen_permute import DataHandler,testDataHandler
dh=DataHandler(batch_size)
dh_test=testDataHandler(batch_size)
x,y=dh.get_batch()
seq_len=x.shape[1]
feature_size=x.shape[2]

U_bound=pow(args.MAG, 1.0 / seq_len)
U_lowbound=pow(1.0/args.MAG, 1.0 / seq_len)
act=rectify
lr=args.lr
num_layers=args.num_layers
opti=lasagne.updates.adam


def build_lstm_network(rnnmodel):
    net = {}    
    net['input'] = InputLayer((batch_size, seq_len, feature_size))
    net['rnn']=rnnmodel(net['input'],hidden_units,forgetgate=lasagne.layers.Gate(b=lasagne.init.Constant(1.)),peepholes=False, only_return_final=True,grad_clipping=args.gradclipvalue)
    net['out']=DenseLayer(net['rnn'],outputclass,nonlinearity=softmax)
    return net  
def build_rnn_network(rnnmodel):
    net = {}    
    net['input'] = InputLayer((batch_size, seq_len, feature_size))
    net['rnn']=rnnmodel(net['input'],hidden_units,nonlinearity=act,W_in_to_hid=Normal(args.ini),W_hid_to_hid=lambda shape:  np.identity(hidden_units,dtype=np.float32),only_return_final=True ,grad_clipping=args.gradclipvalue)
    net['out']=DenseLayer(net['rnn'],outputclass,nonlinearity=softmax)
    return net


ini_W=HeNormal(gain=np.sqrt(2)/2.0)
if args.use_bn_afterrnn:
  ini_W=Uniform(args.ini)


def build_res_rnn_network(rnnmodel):
    net = {}        
    net['input'] = InputLayer((batch_size, seq_len, feature_size))     
    net['rnn0']=DimshuffleLayer(net['input'],(1,0,2))
    for l in range(1, num_layers+1):
      hidini=0
      if l==num_layers:
        hidini=U_lowbound

      net['rnn%d'%(l-1)]=ReshapeLayer(net['rnn%d'%(l-1)], (batch_size* seq_len, -1))          
      net['rnn%d'%(l-1)]=DenseLayer(net['rnn%d'%(l-1)],hidden_units,W=ini_W,b=Uniform(range=(0,args.ini_b)),nonlinearity=None)  #W=Uniform(ini_rernn_in_to_hid),         #
      net['rnn%d'%(l-1)]=ReshapeLayer(net['rnn%d'%(l-1)], (seq_len, batch_size,  -1))  
      
      net['rnn%d'%l]=net['rnn%d'%(l-1)]
      if not args.use_bn_afterrnn:
        net['rnn%d'%l]=BatchNormLayer(net['rnn%d'%l],axes= (0,1),beta=Uniform(range=(0,args.ini_b)))       
      
      net['rnn%d'%l]=rnnmodel(net['rnn%d'%l],hidden_units,W_hid_to_hid=Uniform(range=(hidini,U_bound)),nonlinearity=act,only_return_final=False, grad_clipping=args.gradclipvalue)      
      if args.use_bn_afterrnn:
        net['rnn%d'%l]=BatchNormLayer(net['rnn%d'%l],axes= (0,1))
      if l==num_layers:  
        net['rnn%d'%num_layers]=lasagne.layers.SliceLayer(net['rnn%d'%num_layers],indices=-1, axis=0)     
           
    net['out']=DenseLayer(net['rnn%d'%num_layers],outputclass,nonlinearity=softmax)
    return net


if args.model=='rnn':
  learn_net=build_rnn_network(RecurrentLayer)
elif args.model=='lstm':
  learn_net=build_lstm_network(LSTMLayer)
elif args.model=='indrnn':
  learn_net=build_res_rnn_network(indrnn_onlyrecurrent)  


X_sym = T.tensor3('inputs',dtype=theano.config.floatX)
y_sym = T.ivector()#T.vector('label',dtype=theano.config.floatX)    
   
prediction = lasagne.layers.get_output(learn_net['out'], X_sym,deterministic=False)#,batch_norm_use_averages=True
loss = T.mean(lasagne.objectives.categorical_crossentropy(prediction, y_sym))
acc=T.mean(lasagne.objectives.categorical_accuracy(prediction, y_sym, top_k=1),dtype=theano.config.floatX)
if args.use_weightdecay_nohiddenW:
  params = lasagne.layers.get_all_params(learn_net['out'], regularizable=True)
  for para in params:
    if para.name!='hidden_to_hidden.W':
      loss += args.decayrate *lasagne.regularization.apply_penalty(para, lasagne.regularization.l2)#*T.clip(T.abs_(para)-1,0,100))  


params = lasagne.layers.get_all_params(learn_net['out'], trainable=True)
  
learning_ratetrain = T.scalar(name='learning_ratetrain',dtype=theano.config.floatX)

grads = theano.grad(loss, params)
  
updates = opti( grads, params, learning_rate=learning_ratetrain)#nesterov_momentum( loss, params, learning_rate=learning_ratetrain)#

print('Compiling')
train_fn = theano.function([X_sym, y_sym,learning_ratetrain], [loss,acc], updates=updates)

test_prediction = lasagne.layers.get_output(learn_net['out'], X_sym,deterministic=True,batch_norm_use_averages=False)
test_loss = T.mean(lasagne.objectives.categorical_crossentropy(test_prediction, y_sym))
test_acc=T.mean(lasagne.objectives.categorical_accuracy(test_prediction, y_sym, top_k=1),dtype=theano.config.floatX)
test_fn = theano.function([X_sym, y_sym], [test_loss,test_acc])

bn_test_prediction = lasagne.layers.get_output(learn_net['out'], X_sym,deterministic=True)
bn_test_loss = T.mean(lasagne.objectives.categorical_crossentropy(bn_test_prediction, y_sym))
bn_test_acc=T.mean(lasagne.objectives.categorical_accuracy(bn_test_prediction, y_sym, top_k=1),dtype=theano.config.floatX)
bn_test_fn = theano.function([X_sym, y_sym], [bn_test_loss,bn_test_acc])

      

learning_rate=np.float32(lr)
print ('learning rate',learning_rate)
tacc=0
count=0
for batchi in range(1,10000000):
  x,y=dh.get_batch()
  
  if args.model=='indrnn':
    i=0
    for para in params:
      if para.name=='hidden_to_hidden.W':
        para.set_value(np.clip(para.get_value(),-1*U_bound,U_bound))          
      i+=1   

  mse,acc=train_fn(x, y,learning_rate)
  tacc+=acc
  count+=1
  
  if batchi%1000==0:#1000
    print ('train acc',tacc/count)
    count=0
    tacc=0

    totaltestacc=0
    totatltestno=0
    #learning_ratetrainbase=learning_ratetrainbase*(1 - 1e-7)
    while(1):
      inputs, targets = dh_test.get_batch()
      test_mse,test_acc = test_fn(inputs,targets)
      totaltestacc+=test_acc
      totatltestno+=1
      if totatltestno==dh_test.GetDatasetSize():
        break
    print ("accuracy: ", totaltestacc/totatltestno)   

    totaltestacc=0
    totatltestno=0
    while(1):
      inputs, targets = dh_test.get_batch()
      test_mse,test_acc = bn_test_fn(inputs,targets)
      totaltestacc+=test_acc
      totatltestno+=1
      if totatltestno==dh_test.GetDatasetSize():
        break
    print ("bn_accuracy: ", totaltestacc/totatltestno)   
      
  if batchi%(100*6000)==0:    #dh.GetDatasetSize()==0:
    learning_rate=np.float32(learning_rate*0.1)
    print ('learning rate',learning_rate)                
    if learning_rate<1e-8:
      break
      
save_name='MNIST_'+args.model    
np.savez(save_name, *lasagne.layers.get_all_param_values(learn_net['out']))