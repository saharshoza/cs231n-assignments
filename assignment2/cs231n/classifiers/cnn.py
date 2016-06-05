import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.params['W1'] = np.random.normal(0,weight_scale,(num_filters,input_dim[0],filter_size,filter_size))
    self.params['b1'] = np.zeros((num_filters,))
    self.params['W2'] = np.random.normal(0,weight_scale,(num_filters*(((input_dim[1] - pool_param['pool_height'])/pool_param['stride']) + 1)*(((input_dim[2] - pool_param['pool_width'])/pool_param['stride']) + 1),hidden_dim))
    self.params['b2'] = np.zeros((hidden_dim,))
    self.params['W3'] = np.random.normal(0,weight_scale,(hidden_dim,num_classes))
    self.params['b3'] = np.zeros((num_classes,))   
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    num_filters = W1.shape[0]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    conv_pool_out, conv_pool_cache = conv_relu_pool_forward(X,self.params['W1'],self.params['b1'],conv_param,pool_param)
    hidden_out, hidden_cache = affine_relu_forward(conv_pool_out , self.params['W2'], self.params['b2'])
    scores, scores_cache = affine_forward(hidden_out, self.params['W3'], self.params['b3'])

    scores_exp = np.exp(scores)
    scores_normalised = (scores_exp.T/np.sum(scores_exp,axis=1)).T
    loss = -np.sum(np.log(scores_normalised[xrange(scores_normalised.shape[0]),y]))/float(scores_normalised.shape[0])
    loss = loss + 0.5*self.reg*(np.sum(self.params['W1']*self.params['W1']) + np.sum(self.params['W2']*self.params['W2']) + np.sum(self.params['W3']*self.params['W3']))

    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    grads = {}
    d_scores = scores_normalised
    d_scores[xrange(scores_normalised.shape[0]),y] -= 1
    d_scores = d_scores/float(d_scores.shape[0])
    dx_hidden, grads['W3'], grads['b3'] = affine_backward(d_scores, scores_cache)
    grads['W3'] += self.reg*self.params['W3']
    dx_pool, grads['W2'], grads['b2'] = affine_relu_backward(dx_hidden, hidden_cache)
    grads['W2'] += self.reg*self.params['W2']
    dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(dx_pool, conv_pool_cache)
    grads['W1'] += self.reg*self.params['W1']
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
pass
