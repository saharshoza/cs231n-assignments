import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  x_mat = x.reshape(x.shape[0],-1)
  #f = lambda x: x.clip(min=0)
  #out = f(np.dot(x_mat,w)+b)
  out = np.dot(x_mat,w) + b
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  x_mat = x.reshape(x.shape[0],-1)
  num_examples = float(dout.shape[0])
  db = np.sum(dout,axis=0)
  #dout[affine_forward(x,w,b)[0] <= 0] = 0 
  dw = np.dot(dout.transpose(),x_mat).transpose()
  dx_mat = np.dot(dout,w.transpose())
  dx = dx_mat.reshape(x.shape)
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  out = x.clip(min=0)
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  dx = dout
  #dx[relu_forward(x)[0] <= 0] = 0
  dx[x <= 0] = 0
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  x = x.reshape(x.shape[0],-1)
  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    mu = np.mean(x,axis=0)
    xmu = x - mu
    sq = xmu**2
    var = np.mean(sq,axis=0)
    sqrtvar = (var+eps)**0.5
    ivar = 1./sqrtvar
    xhat = xmu*ivar
    gammax = xhat*gamma
    out = gammax + beta
    sample_mean = np.mean(mu)
    sample_var = np.mean(var)
    running_mean = (momentum*running_mean) + ((1-momentum)*sample_mean)
    running_var = (momentum*running_var) + ((1-momentum)*sample_var)
    cache = (xhat,gamma,xmu,ivar,sqrtvar,var,eps)
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    sample_mean = bn_param['running_mean']
    sample_var = bn_param['running_var']
    out = gamma*((x - sample_mean)/sample_var) + beta
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache
  dout = dout.reshape(dout.shape[0],-1)
  dbeta = np.sum(dout,axis=0)
  dgammax = dout
  dgamma = np.sum((dgammax*xhat), axis=0)
  dxhat = dgammax*gamma
  divar = np.sum(dxhat*xmu,axis=0)
  dxmu1 = dxhat*ivar
  dsqrtvar = -(1./(sqrtvar**2))*divar
  dvar = (0.5)*(1./((var + eps)**0.5))*dsqrtvar
  dsq = dvar*(1./dout.shape[0])*np.ones(dout.shape)
  dxmu2 = 2*dsq*xmu
  dx1 = dxmu1 + dxmu2
  dmu = -np.sum((dxmu2+dxmu1),axis=0)
  dx2 = dmu*(1./dout.shape[0])*np.ones(dout.shape)
  dx = dx1 + dx2
#  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    mask = ((np.random.randn(*x.shape) > p)/p)
    out = mask*x
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    pass
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    out = x
    mask = None
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    pass
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    dx = mask*dout
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    pass
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  x_pad = np.pad(x,((0,0),(0,0),(conv_param['pad'],conv_param['pad']),(conv_param['pad'],conv_param['pad'])),'constant',constant_values=(0))
  N = x_pad.shape[0]
  F = w.shape[0]
  out_size = (N, F, ((x_pad.shape[2] - w.shape[2])/conv_param['stride']+1), ((x_pad.shape[3] - w.shape[3])/conv_param['stride']+1))
  out = np.zeros(out_size)
  w_col = w.reshape(w.shape[0],-1)
  x_col = np.zeros((w.shape[2]*w.shape[3]*w.shape[1],out_size[2]*out_size[3]*N))

  #x_flat = x.reshape(x.shape[0],-1)
  block_width = w.shape[3]
  block_height = w.shape[2]
  h_stride_iter = 0
  v_stride_iter = 0
  num_example = 0
  #print x.shape
  #print w.shape
  for col_iter in range(0,N*out_size[2]*out_size[3]):

    if block_height == x_pad.shape[2] and block_width > x_pad.shape[3]:
      num_example += 1
      h_stride_iter = 0
      v_stride_iter = 0
      block_height = w.shape[2]
      block_width = w.shape[3]
      #print 'next example'
    elif block_width > x_pad.shape[3]:
      block_width = w.shape[3]
      h_stride_iter = 0
      v_stride_iter += 1
      block_height += conv_param['stride']
      #print 'move down'

    #print col_iter
    #print block_width
    #print block_height

    x_col[:,col_iter] = x_pad[num_example,:, v_stride_iter*conv_param['stride']: block_height ,h_stride_iter*conv_param['stride'] : block_width].reshape(-1)
    block_width += conv_param['stride']
    h_stride_iter += 1
    #print x_col
    
  out_col = (np.dot(w_col,x_col).T + b).T

  for num_example in range(0,N):
    out[num_example] = out_col[:,(num_example*out_col.shape[1]/N) : (out_col.shape[1]/N)*(1+num_example)].reshape(out_size[1],out_size[2], out_size[3])
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x_col, w_col, x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """

  dx, dw, db = None, None, None
  N,F,HH_out,WW_out = dout.shape
  x_col, w_col, x, w, b, conv_param = cache
  _,C,H,W = x.shape
  _,_,HH,WW = w.shape
  dout_col = np.zeros((F,N*HH_out*WW_out))
  dout_flip = np.zeros(dout.shape)
  dx = np.zeros((N,C,H,W))
  W += 2*conv_param['pad']
  H += 2*conv_param['pad']
  dx_pad = np.zeros((N,C,H,W))
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################

  db = np.sum(np.sum(np.sum(dout,axis=3),axis=2),axis=0)
  
  for i in range(0,dout.shape[1]):
    dout_col[i] = dout[:,i,:,:].reshape(-1)

  dw_col = np.dot(dout_col,x_col.T)
  dx_col = np.dot(w_col.T,dout_col)

  dw = dw_col.reshape(F,C,HH,WW)

  num_example = 0
  h_stride_iter = 0
  v_stride_iter = 0
  block_width = 0
  block_height = 0

  for i in range(0,dout_col.shape[1]):
    if block_width >= W:
      h_stride_iter = 0
      v_stride_iter += conv_param['stride']
    if block_width >= W and block_height >= H:
      h_stride_iter = 0
      v_stride_iter = 0
      num_example += 1

    block_width = h_stride_iter + WW
    block_height = v_stride_iter + HH
    dx_pad[num_example , : , v_stride_iter : block_height, h_stride_iter : block_width] += dx_col[:,i].reshape(C,HH,WW)
    h_stride_iter += conv_param['stride']

  dx = dx_pad[:,:,conv_param['pad']:H - conv_param['pad'], conv_param['pad'] : W - conv_param['pad']]

  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None

  pool_height = pool_param['pool_height'] 
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']
  N,C,H,W = x.shape

  out = np.zeros((N,C,((H-pool_height)/stride + 1),((W-pool_width)/stride + 1)))
  out_index = np.zeros(out.shape)
  x_reshape = np.zeros((N,((H-pool_height)/stride + 1)*((W-pool_width)/stride + 1),C,pool_height,pool_width))
  x_reshape_index = np.zeros(x_reshape.shape)
  x_index = np.zeros(x.shape)
  x_index[:,:,:] = np.array(range(0,(H*W))).reshape(H,W)

  width_limit = pool_width
  height_limit = pool_height
  reshape_iter = 0
  width_iter = 0
  height_iter = 0
  num_example = 0

  for i in range(0,x_reshape.shape[1]*x_reshape.shape[0]):
    if width_limit == W:
      width_iter = 0
      height_iter += 1
    if height_limit == H and width_limit == W:
      num_example += 1
      reshape_iter = 0
      width_iter = 0
      height_iter = 0
    
    width_limit = (width_iter)*stride + pool_width
    height_limit = ((height_iter)*stride) + pool_height
    x_reshape[num_example,reshape_iter,:,:,:] = x[num_example,:, (height_iter*stride): ((height_iter)*stride) + pool_height,  width_iter*stride : (width_iter)*stride + pool_width]
    x_reshape_index[num_example,reshape_iter,:,:,:] = x_index[num_example,:, (height_iter*stride): ((height_iter)*stride) + pool_height,  width_iter*stride : (width_iter)*stride + pool_width]
    reshape_iter += 1
    width_iter += 1

  #print x_reshape

  x_max_inter = np.max(x_reshape,axis=4)
  
  x_max_inter_arg = np.argmax(x_reshape,axis=4)
  x_inter_index = np.zeros(x_max_inter_arg.shape)

  for num in range(0,N):
    for split_iter in range(0,x_reshape.shape[1]):
      for row_iter in range(0,x_max_inter_arg.shape[3]):
        x_inter_index[num,split_iter,:,row_iter] = x_reshape_index[num,split_iter,tuple(range(0,C)),row_iter,x_max_inter_arg[num,split_iter,:,row_iter]]

  #print x_max_inter_arg
  #print x_inter_index

  x_max =  np.max(x_max_inter,axis=3)
  x_max_arg = np.argmax(x_max_inter,axis=3)

  x_max_index = np.zeros(x_max_arg.shape)

  #print x_max_arg.shape

  for num in range(0,N):
    for split_iter in range(0,x_reshape.shape[1]):
      x_max_index[num,split_iter,:] = x_inter_index[num,split_iter,tuple(range(0,C)),x_max_arg[num,split_iter,:]]
  
  #print x_max_index

  width_iter = 0
  height_iter = 0
  row_iter = 0
  num_example = 0
  for i in range(0,x_max.shape[0]*x_max.shape[1]):
    if width_iter >= out.shape[3]:
      width_iter = 0
      height_iter += 1
    if row_iter >= x_max.shape[1]:
      row_iter = 0
      num_example += 1
      width_iter = 0
      height_iter = 0

    out[num_example,:,height_iter,width_iter] = x_max[num_example,row_iter]
    out_index[num_example, :, height_iter, width_iter] = x_max_index[num_example, row_iter]
    row_iter += 1
    width_iter += 1
  #print out_index
  #print out
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param, out_index, x_index)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param, out_index, x_index = cache
  N,C,H,W = x.shape
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  dx = np.zeros(x.shape)
  for num_example in range(0,N):
    for channel in range(0,C):
      for row_iter in range(0,dout.shape[2]):
        for col_iter in range(0,dout.shape[3]):
          height_ind, width_ind = np.where(x_index[num_example,channel,:,:] == out_index[num_example,channel,row_iter,col_iter])
          dx[num_example, channel, height_ind, width_ind] += dout[num_example,channel,row_iter,col_iter]
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = np.zeros(x.shape), []
  N, C, H, W = x.shape
  for i in range(0,C):
    out_flat, cache_channel = batchnorm_forward(x[:,i,:,:],gamma[i],beta[i],bn_param)
    out[:,i,:,:] = out_flat.reshape(N,H,W)
    cache.append(cache_channel)

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  N, C, H, W = dout.shape
  dgamma = np.ones(C)
  dbeta = np.ones(C)
  dx = np.zeros(dout.shape)
  for i in range(0,C):
    out_channel, gamma_channel, beta_channel = batchnorm_backward(dout[:,i,:,:],cache[i])
    dgamma[i] = np.sum(gamma_channel)
    dbeta[i] = np.sum(beta_channel)
    dx[:,i,:,:] = out_channel.reshape(N,H,W)

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
