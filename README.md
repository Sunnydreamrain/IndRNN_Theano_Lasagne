# IndRNN (Theano+Lasagne)
This code is to implement the [IndRNN](https://arxiv.org/abs/1803.04831). It is based on Theano and Lasagne.

Please cite the following paper if you find it useful.  
[Shuai Li, Wanqing Li, Chris Cook, Ce Zhu, and Yanbo Gao. "Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN." CVPR 2018.](https://arxiv.org/abs/1803.04831)

@article{li2018independently,  
  title={Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN},  
  author={Li, Shuai and Li, Wanqing and Cook, Chris and Zhu, Ce and Gao, Yanbo},  
  booktitle={CVPR2018},  
  year={2018}  
} 

# Usuage 
`IndRNN.py` provides the IndRNN function as described in the paper.  
`IndRNN_onlyrecurrent.py` provides only the recurrent+activation of the IndRNN function. Therefore, processing of the input with dense connection or convolution operation is needed. This is usedful for adding batch normalization (BN) between the processing of input and activation function.

## For the adding example:   
`python -u adding.py`  
Different options are available in adding.py.  
Example: `python -u adding.py --model indrnn --seq_len 100`  
Example of using GPU: `THEANO_FLAGS='floatX=float32,device=cuda0,mode=FAST_RUN' python -u adding.py --model indrnn --seq_len 100`  

## For the pixel MNIST example:  
`python -u pixelmnist.py --use_bn_afterrnn`   
or with options: 
`python -u adding.py --model indrnn --num_layers 6 --hidden_units 128 --use_bn_afterrnn`  
Example of using GPU: `THEANO_FLAGS='floatX=float32,device=cuda0,mode=FAST_RUN' python -u adding.py --model indrnn --num_layers 6 --hidden_units 128 --use_bn_afterrnn`  

For this task, the batch normalization (BN) is used. It can be used before the activation function (relu) or after it. In our experiments, it converges faster by putting BN after the activation function.  

## Other tasks will come soon.
