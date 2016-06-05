import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
 # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  loss = 0.0
  dW = np.zeros_like(W)
  num_trains = X.shape[0]

  margin_mat = np.exp(np.dot(X,W).transpose())
  margin_denominator = np.sum(margin_mat,axis=0)
  margin_normalized = margin_mat/margin_denominator
  loss_mat = -np.log(margin_normalized)
  loss_array = loss_mat[y,xrange(loss_mat.shape[1])]
  loss = np.sum(loss_array)
  loss /= float(num_trains)
  loss += 0.5*reg*np.sum(W*W)

  grad_compute_mat = margin_normalized*(-1)
  grad_compute_mat[y,xrange(grad_compute_mat.shape[1])] = 1 + grad_compute_mat[y,xrange(grad_compute_mat.shape[1])]
  dW = (-1)*np.dot(grad_compute_mat,X).transpose()
  dW /= float(num_trains)



  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
 # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  loss = 0.0
  dW = np.zeros_like(W)
  num_trains = X.shape[0]

  margin_mat = np.exp(np.dot(X,W).transpose())
  margin_denominator = np.sum(margin_mat,axis=0)
  margin_normalized = margin_mat/margin_denominator
  loss_mat = -np.log(margin_normalized)
  loss_array = loss_mat[y,xrange(loss_mat.shape[1])]
  loss = np.sum(loss_array)
  loss /= float(num_trains)
  loss += 0.5*reg*np.sum(W*W)

  grad_compute_mat = margin_normalized*(-1)
  grad_compute_mat[y,xrange(grad_compute_mat.shape[1])] = 1 + grad_compute_mat[y,xrange(grad_compute_mat.shape[1])]
  dW = (-1)*np.dot(grad_compute_mat,X).transpose()
  dW /= float(num_trains)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

