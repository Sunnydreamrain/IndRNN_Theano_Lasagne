from __future__ import print_function
import sys
import argparse
import os

import time

import lasagne
import theano
import numpy as np
import theano.tensor as T

from lasagne.layers import InputLayer,ReshapeLayer,DimshuffleLayer,Gate
from lasagne.layers import Conv2DLayer
from lasagne.layers import Pool2DLayer
from lasagne.layers import DenseLayer

from lasagne.layers import ConcatLayer,NonlinearityLayer,DropoutLayer
from lasagne.layers.normalization import batch_norm
from lasagne.nonlinearities import softmax, rectify,tanh

from lasagne.layers import RecurrentLayer,LSTMLayer
from IndRNN import IndRNNLayer as indrnn



parser = argparse.ArgumentParser(description='IndRNN solving the adding problem')
parser.add_argument('--model', type=str, default='indrnn', help='models')
parser.add_argument('--hidden_units', type=int, default=128, help='humber of hidden units per layer')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seq_len', type=int, default=500)
parser.add_argument('--MAG', type=int, default=2)
parser.add_argument('--act', type=str, default='relu')
parser.add_argument('--lr', type=np.float32, default=2e-4, help='lr')
parser.add_argument('--in2hidW', type=np.float32, default=0.001)
parser.add_argument('--gradclipvalue', type=np.float32, default=10)
args = parser.parse_args()

print(args)

batch_size = args.batch_size
seq_len=args.seq_len
hidden_units=args.hidden_units
feature_size=2
outputclass=1


U_bound=pow(args.MAG, 1.0 / seq_len)
U_lowbound=pow(1.0/args.MAG, 1.0 / seq_len)
if args.act=='tanh':
  U_bound=pow(args.MAG/(pow(0.9,seq_len/10.0)), 1.0 / seq_len)

act=rectify
if args.act=='tanh':
  act=tanh

lr=args.lr
opti=lasagne.updates.adam#rmsprop




def build_rnn_network(rnnmodel):
    net = {}    
    net['input'] = InputLayer((batch_size, seq_len, feature_size))
    if rnnmodel==LSTMLayer:
      net['rnn']=rnnmodel(net['input'],hidden_units,forgetgate=lasagne.layers.Gate(b=lasagne.init.Constant(1.)),only_return_final=True,grad_clipping=args.gradclipvalue )
    elif act==rectify:
      net['rnn']=rnnmodel(net['input'],hidden_units,W_in_to_hid=lasagne.init.Normal(args.in2hidW), W_hid_to_hid=lambda shape: np.identity(hidden_units,dtype=np.float32),nonlinearity=act,only_return_final=True,grad_clipping=args.gradclipvalue )
    elif act==tanh:
      net['rnn']=rnnmodel(net['input'],hidden_units,W_in_to_hid=lasagne.init.Normal(args.in2hidW), nonlinearity=act,only_return_final=True,grad_clipping=args.gradclipvalue )
    net['out']=DenseLayer(net['rnn'],outputclass,nonlinearity=None)
    print (lasagne.layers.get_output_shape(net['out']))
    return net


  
def build_indrnn_network(res_rnnmodel):
    net = {}    
    net['input'] = InputLayer((batch_size, seq_len, feature_size))
    if act==rectify:
      net['rnn0']=res_rnnmodel(net['input'],hidden_units,W_in_to_hid=lasagne.init.Normal(args.in2hidW), nonlinearity=act,W_hid_to_hid=lasagne.init.Uniform(range=(0,U_bound)),grad_clipping=args.gradclipvalue)
      net['rnn']=res_rnnmodel(net['rnn0'],hidden_units,W_in_to_hid=lasagne.init.Normal(args.in2hidW), nonlinearity=act,W_hid_to_hid=lasagne.init.Uniform(range=(U_lowbound,U_bound)),only_return_final=True,grad_clipping=args.gradclipvalue)
    elif act==tanh:
      net['rnn0']=res_rnnmodel(net['input'],hidden_units,W_in_to_hid=lasagne.init.Normal(args.in2hidW), nonlinearity=act,W_hid_to_hid=lasagne.init.Uniform(range=(U_bound)),grad_clipping=args.gradclipvalue)
      net['rnn']=res_rnnmodel(net['rnn0'],hidden_units,W_in_to_hid=lasagne.init.Normal(args.in2hidW), nonlinearity=act,W_hid_to_hid=lasagne.init.Uniform(range=(U_bound)),only_return_final=True,grad_clipping=args.gradclipvalue)
    net['out']=DenseLayer(net['rnn'],outputclass,nonlinearity=None)
    return net  
  

def generate_data(time_steps, n_data):
    x = np.asarray(np.zeros((time_steps, int(n_data), 2)),
                   dtype=theano.config.floatX)

    x[:,:,0] = np.asarray(np.random.uniform(low=0.,
                                            high=1.,
                                            size=(time_steps, n_data)),
                          dtype=theano.config.floatX)
    
    

    inds = np.asarray(np.random.randint(time_steps//2, size=(n_data, 2)))
    inds[:, 1] += time_steps//2  
    
    for i in range(int(n_data)):
        x[inds[i, 0], i, 1] = 1.0
        x[inds[i, 1], i, 1] = 1.0
 
    y = (x[:,:,0] * x[:,:,1]).sum(axis=0)
    y = np.reshape(y, (n_data, 1))
    x=np.transpose(x, (1, 0, 2))

    return x, y


 
if args.model=='rnn':
  learn_net=build_rnn_network(RecurrentLayer)
elif args.model=='lstm':
  learn_net=build_rnn_network(LSTMLayer)
elif args.model=='indrnn':
  learn_net=build_indrnn_network(indrnn)  

  
X_sym = T.tensor3('inputs',dtype=theano.config.floatX)
y_sym = T.matrix('label',dtype=theano.config.floatX)    
   
prediction = lasagne.layers.get_output(learn_net['out'], X_sym,deterministic=False)
loss = lasagne.objectives.squared_error(prediction, y_sym).mean()

params = lasagne.layers.get_all_params(learn_net['out'], trainable=True)

learning_ratetrain = T.scalar(name='learning_ratetrain',dtype=theano.config.floatX)

grads = theano.grad(loss, params)
#grads = [T.clip(g, -1, 1) for g in grads] 
updates = opti( grads, params, learning_rate=learning_ratetrain)#nesterov_momentum( loss, params, learning_rate=learning_ratetrain)#
print('Compiling')
train_fn = theano.function([X_sym, y_sym,learning_ratetrain], [loss,], updates=updates)
test_fn = theano.function([X_sym, y_sym], [loss,])
      

learning_rate=np.float32(lr)
tmse=0
lastmse=100
count=0
for batchi in range(1,10000000):
  x,y=generate_data(seq_len, batch_size)
  
  if args.model=='indrnn':
    i=0
    for para in params:
      if para.name=='hidden_to_hidden.W':
        para.set_value(np.clip(para.get_value(),-1*U_bound,U_bound))           
      i+=1  
    
  mse,=train_fn(x, y,learning_rate)
  if np.isnan(mse):
    print ('NaN detected in cost')
    assert(2==3)
  if np.isinf(mse):
    print ('INF detected in cost')
    assert(2==3)  
  tmse+=mse
  
  if batchi%100==0:
    print ('training', tmse/100.0)
    count+=1
    
    x,y=generate_data(seq_len, 1000)
    mse,=test_fn(x, y)
    print ('accuracy:', mse)
    
    if (count>200):
      learning_rate=np.float32(learning_rate*0.1)
      print ('learning rate',learning_rate)
      count=0      
      if learning_rate<1e-6:
        break

    tmse=0

save_name=args.model+str(seq_len)
np.savez(save_name, *lasagne.layers.get_all_param_values(learn_net['out']))
