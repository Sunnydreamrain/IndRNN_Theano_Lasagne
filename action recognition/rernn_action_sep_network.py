from __future__ import print_function
import sys
import argparse
import os
import time
import theano
import numpy as np
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer,ReshapeLayer,DimshuffleLayer,Gate,BatchNormLayer
from lasagne.layers import DenseLayer,ElemwiseSumLayer,SliceLayer
from lasagne.layers import ConcatLayer,NonlinearityLayer,DropoutLayer
from lasagne.nonlinearities import softmax, rectify,tanh
from lasagne.init import Uniform,Normal,HeNormal

from IndRNN_onlyrecurrent import IndRNNLayer_onlyrecurrent as indrnn_onlyrecurrent

import opts

sys.setrecursionlimit(10000)
parser = argparse.ArgumentParser(description='network')

#parser.set_defaults(use_weightdecay_nohiddenW=True, use_bn=True, use_birnn=True)

opts.train_opts(parser)
args = parser.parse_args()

batch_size = args.batch_size
seq_len=args.seq_len
num_layers=args.num_layers
hidden_units=args.hidden_units
outputclass=60
indim=50#150
droprate=args.droprate
gradclipvalue=10
act=rectify
U_bound=pow(args.MAG, 1.0 / seq_len)
U_lowbound=pow(1.0/args.MAG, 1.0 / seq_len)
if args.bn_drop:
  from batch_norm_withdrop_timefirst import BatchNormLayer as dropBatchNormLayer
  
rnnmodel=indrnn_onlyrecurrent


ini_W=HeNormal(gain=np.sqrt(2)/np.sqrt(args.seq_len))
if args.bn_drop or args.use_bn_afterrnn:
  ini_W=Uniform(args.ini_in2hid)
  
def build_indrnn_network(X_sym):
    net = {}        
    net['input0'] = InputLayer((batch_size, seq_len, indim, 3),X_sym)
    net['input']=ReshapeLayer(net['input0'], (batch_size, seq_len, indim*3))    
    net['rnn0']=DimshuffleLayer(net['input'],(1,0,2))
    for l in range(1, num_layers+1):
      hidini=0
      if l==num_layers:
        hidini=U_lowbound
      net['rnn%d'%(l-1)]=ReshapeLayer(net['rnn%d'%(l-1)], (batch_size* seq_len, -1))                
      net['rnn%d'%(l-1)]=DenseLayer(net['rnn%d'%(l-1)],hidden_units,W=ini_W,b=lasagne.init.Constant(args.ini_b),nonlinearity=None)         #
      net['rnn%d'%(l-1)]=ReshapeLayer(net['rnn%d'%(l-1)], (seq_len, batch_size,  -1))  
      if args.conv_drop:
        net['rnn%d'%(l-1)]=DropoutLayer(net['rnn%d'%(l-1)], p=droprate, shared_axes=(0,))    
      net['rnn%d'%l]=net['rnn%d'%(l-1)]
      if not args.bn_drop and not args.use_bn_afterrnn:
        net['rnn%d'%l]=BatchNormLayer(net['rnn%d'%l],beta=lasagne.init.Constant(args.ini_b),axes= (0,1))    
               
      net['rnn%d'%l]=rnnmodel(net['rnn%d'%l],hidden_units,W_hid_to_hid=Uniform(range=(hidini,U_bound)),nonlinearity=act,only_return_final=False, grad_clipping=gradclipvalue)
                         
      if args.use_bn_afterrnn:
        net['rnn%d'%l]=BatchNormLayer(net['rnn%d'%l],axes= (0,1))
      if args.bn_drop:
        net['rnn%d'%l]=dropBatchNormLayer(net['rnn%d'%l],axes= (0,1),droprate=droprate)
      if args.use_dropout and l%args.drop_layers==0:
        net['rnn%d'%l]=DropoutLayer(net['rnn%d'%l], p=droprate, shared_axes=(0,))        
        
    net['rnn%d'%num_layers]=lasagne.layers.SliceLayer(net['rnn%d'%num_layers],indices=-1, axis=0)      
    net['out']=DenseLayer(net['rnn%d'%num_layers],outputclass,nonlinearity=softmax)
    return net