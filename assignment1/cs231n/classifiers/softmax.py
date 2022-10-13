from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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
    
    num_train=X.shape[0]
    num_class=W.shape[1]
    
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    score=X.dot(W)
    
    for i in range(num_train):
        score_shift=max(score[i])
        score[i]-=score_shift
        score_e=np.exp(score[i])
        
        loss_i=-score[i,y[i]]+np.log(sum(score_e))                   
        loss+=loss_i
        
        for j in range(num_class):
            p=np.exp(score[i,j])/sum(np.exp(score[i]))
            if(j==y[i]):
                dW[:,j]+=p*X[i]-X[i]
            else:
                dW[:,j]+=p*X[i]
                          
    loss/=num_train
    loss+=0.5*reg*np.sum(W*W)
    dW=dW/num_train+reg*W
                          
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train=X.shape[0]
    num_class=W.shape[1]

    score=X.dot(W)
    score_shift=np.max(X,axis=1).reshape(num_train,-1)
    score-=score_shift
    loss=-sum(y)+sum(np.log(np.sum(np.exp(score),axis=1)))
    
    softmax_output=np.exp(score)/np.sum(np.exp(score),axis=1).reshape((-1,1))
    dS=softmax_output.copy()
    dS[range(num_train),list(y)]+=-1
    dW=(X.T).dot(dS)
    dW=dW/num_train+reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
