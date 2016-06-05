import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    self.params['W1'] = np.random.normal(0,weight_scale,input_dim*hidden_dim).reshape(input_dim,hidden_dim)
    self.params['b1'] = np.zeros((hidden_dim,))
    self.params['W2'] = np.random.normal(0,weight_scale,hidden_dim*num_classes).reshape(hidden_dim,num_classes)
    self.params['b2'] = np.zeros((num_classes,))
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    grads = {'W1': None,
             'b1': None,
             'W2': None,
             'b2': None}
    
    affine1_out, affine1_cache  = affine_forward(X,self.params['W1'],self.params['b1'])
    relu1_out, relu1_cache = relu_forward(affine1_out)
    affine2_out, affine2_cache = affine_forward(relu1_out,self.params['W2'],self.params['b2'])
    scores, relu2_cache = relu_forward(affine2_out)

    scores_exp = np.exp(scores)
    scores_normalised = (scores_exp.T/np.sum(scores_exp,axis=1)).T
    loss = -np.sum(np.log(scores_normalised[xrange(scores_normalised.shape[0]),y]))/float(scores_normalised.shape[0])
    loss = loss + 0.5*self.reg*(np.sum(self.params['W1']*self.params['W1']) + np.sum(self.params['W2']*self.params['W2']))
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    scores_normalised[xrange(scores_normalised.shape[0]),y] -= 1
    dout = scores_normalised/float(scores_normalised.shape[0])
    grad_interim, grads['W2'], grads['b2'] = affine_backward(relu_backward(dout,relu2_cache),affine2_cache)
    _ , grads['W1'], grads['b1'] = affine_backward(relu_backward(grad_interim,relu1_cache),affine1_cache)
    grads['W2'] += self.reg*self.params['W2']
    grads['W1'] += self.reg*self.params['W1']
    loss, grads = loss, grads
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}
    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    self.params['W1'] = np.random.normal(0,weight_scale,input_dim*hidden_dims[0]).reshape(input_dim,hidden_dims[0])
    self.params['b1'] = np.zeros((hidden_dims[0],))

    if self.use_batchnorm:
        self.params['gamma1'] = np.ones(input_dim)
        self.params['beta1'] = np.zeros(input_dim)

    i = 0
    
    for i in range(1,len(hidden_dims)):
    	self.params['W'+str(i+1)] = np.random.normal(0,weight_scale,hidden_dims[i-1]*hidden_dims[i]).reshape(hidden_dims[i-1],hidden_dims[i])
        self.params['b'+str(i+1)] = np.zeros((hidden_dims[i]))
        if self.use_batchnorm:
            self.params['gamma'+str(i+1)] = np.ones(hidden_dims[i-1])
            self.params['beta'+str(i+1)] = np.zeros(hidden_dims[i-1])

    self.params['W'+str(i+2)] = np.random.normal(0,weight_scale,hidden_dims[i]*num_classes).reshape(hidden_dims[i],num_classes)
    self.params['b'+str(i+2)] = np.zeros((num_classes,))

    if use_batchnorm:
        self.params['gamma'+str(i+2)] = np.ones(hidden_dims[i])
        self.params['beta'+str(i+2)] = np.zeros(hidden_dims[i])

    self.hidden_dims = hidden_dims

    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode
    
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    batchnorm_cache = []
    dropout_cache = []
    affine_cache = []
    relu_cache = []
    x_net = X
    for net_iter in range(0,len(self.hidden_dims)):
        if self.use_batchnorm:
            x_net, bn_cache = batchnorm_forward(x_net,self.params['gamma'+str(net_iter+1)] ,self.params['beta'+str(net_iter+1)],bn_param)
            batchnorm_cache.append(bn_cache)
    	affine_out, af_cache = affine_forward(x_net,self.params['W'+str(net_iter+1)],self.params['b'+str(net_iter+1)]) 
        affine_cache.append(af_cache)
        x_net, re_cache = relu_forward(affine_out)
        relu_cache.append(re_cache)
        if self.use_dropout:
            x_net, dp_cache = dropout_forward(x_net,self.dropout_param)
            dropout_cache.append(dp_cache)

    if self.use_batchnorm:
        x_net, bn_cache = batchnorm_forward(x_net,self.params['gamma'+str(net_iter+2)] ,self.params['beta'+str(net_iter+2)],bn_param)
        batchnorm_cache.append(bn_cache)

    scores, af_cache = affine_forward(x_net,self.params['W'+str(net_iter+2)],self.params['b'+str(net_iter+2)])
    affine_cache.append(af_cache)
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    scores_exp = np.exp(scores)
    scores_normalised = (scores_exp.T/np.sum(scores_exp,axis=1)).T
    loss = -np.sum(np.log(scores_normalised[xrange(scores_normalised.shape[0]),y]))/float(scores_normalised.shape[0])
    for reg_iter in range(1,net_iter+2):
    	loss += 0.5*self.reg*np.sum(self.params['W'+str(reg_iter)]*self.params['W'+str(reg_iter)])
    
    grads = {}
    d_scores = scores_normalised
    d_scores[xrange(scores_normalised.shape[0]),y] -= 1
    d_scores = d_scores/float(d_scores.shape[0])
    dx, grads['W'+str(net_iter+2)], grads['b'+str(net_iter+2)] = affine_backward(d_scores,affine_cache[net_iter+1])

    if self.use_batchnorm:
        dx, grads['gamma'+str(net_iter+2)], grads['beta'+str(net_iter+2)] = batchnorm_backward(dx,batchnorm_cache[net_iter+1])

    for i in range(0,net_iter+1):

        grad_iter = net_iter + 1 -i

        if self.use_dropout and self.dropout_param['mode'] == 'train':
            dx = dropout_backward(dx,dropout_cache[grad_iter-1])

        dx = relu_backward(dx,relu_cache[grad_iter-1])

        dx, grads['W'+str(grad_iter)], grads['b'+str(grad_iter)] = affine_backward(dx,affine_cache[grad_iter-1])

        if self.use_batchnorm:
            dx, grads['gamma'+str(grad_iter)], grads['beta'+str(grad_iter)] = batchnorm_backward(dx,batchnorm_cache[grad_iter-1])

 	grads['W'+str(grad_iter)] += self.reg*affine_cache[grad_iter-1][1]
        
    loss, grads = loss, grads
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
