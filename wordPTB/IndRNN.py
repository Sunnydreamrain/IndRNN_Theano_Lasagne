# -*- coding: utf-8 -*-
"""
This code is to implement the IndRNN. The code is based on the Lasagne implementation of RecurrentLayer.

Please cite the following paper if you find it useful.

Shuai Li, Wanqing Li, Chris Cook, Ce Zhu, and Yanbo Gao. "Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN." CVPR 2018.
@article{li2018independently,
  title={Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN},
  author={Li, Shuai and Li, Wanqing and Cook, Chris and Zhu, Ce and Gao, Yanbo},
  booktitle={CVPR2018},
  year={2018}
}
"""
import numpy as np
import theano
import theano.tensor as T
import lasagne.nonlinearities as nonlinearities
import lasagne.init as init
from lasagne.utils import unroll_scan

from lasagne.layers import MergeLayer, Layer
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import CustomRecurrentLayer
import lasagne

__all__ = [
    "MulLayer",
    "IndRNNLayer"
]



class MulLayer(lasagne.layers.Layer):
    def __init__(self, incoming,  W=lasagne.init.Normal(0.01), **kwargs):
        super(MulLayer, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[1]
        self.W = self.add_param(W, (num_inputs, ), name='W')

    def get_output_for(self, input, **kwargs):
        return input * self.W

    def get_output_shape_for(self, input_shape):
        return input_shape#(input_shape[0], self.num_units)


class IndRNNLayer(CustomRecurrentLayer):

    def __init__(self, incoming, num_units,
                 W_in_to_hid=init.Uniform(),
                 W_hid_to_hid=init.Uniform(),
                 b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 **kwargs):

        if isinstance(incoming, tuple):
            input_shape = incoming
        else:
            input_shape = incoming.output_shape
        # Retrieve the supplied name, if it exists; otherwise use ''
        if 'name' in kwargs:
            basename = kwargs['name'] + '.'
            # Create a separate version of kwargs for the contained layers
            # which does not include 'name'
            layer_kwargs = dict((key, arg) for key, arg in kwargs.items()
                                if key != 'name')
        else:
            basename = ''
            layer_kwargs = kwargs
        # We will be passing the input at each time step to the dense layer,
        # so we need to remove the second dimension (the time dimension)
        in_to_hid = DenseLayer(InputLayer((None,) + input_shape[2:]),
                               num_units, W=W_in_to_hid, b=b,
                               nonlinearity=None,
                               name=basename + 'input_to_hidden',
                               **layer_kwargs)
        # The hidden-to-hidden layer expects its inputs to have num_units
        # features because it recycles the previous hidden state
        
        hid_to_hid = MulLayer(InputLayer((None, num_units)),
                                 W=W_hid_to_hid, 
                                name=basename + 'hidden_to_hidden',
                                **layer_kwargs)
#         hid_to_hid = DenseLayer(InputLayer((None, num_units)),
#                                 num_units, W=W_hid_to_hid, b=None,
#                                 nonlinearity=None,
#                                 name=basename + 'hidden_to_hidden',
#                                 **layer_kwargs)

        # Make child layer parameters intuitively accessible
        self.W_in_to_hid = in_to_hid.W
        self.W_hid_to_hid = hid_to_hid.W
        self.b = in_to_hid.b

        # Just use the CustomRecurrentLayer with the DenseLayers we created
        super(IndRNNLayer, self).__init__(
            incoming, in_to_hid, hid_to_hid, nonlinearity=nonlinearity,
            hid_init=hid_init, backwards=backwards, learn_init=learn_init,
            gradient_steps=gradient_steps,
            grad_clipping=grad_clipping, unroll_scan=unroll_scan,
            precompute_input=precompute_input, mask_input=mask_input,
            only_return_final=only_return_final, **kwargs)


