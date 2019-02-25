"""
TODO:
- Move the `negative_sampling` phase in the `train` method for each epochs
"""

from code.utils import sigmoid_array
import numpy as np
from scipy.special import expit

def run_batch(sg, current, contexts_word, negative_samples, w0, w1, epoch_error, sgd_count):
    """
    Run a batch on the `current word` and its `contexts_word` and
    update the weights `w0`, `w1` and the error `epoch_error`, `sgd_count`
    Parameters:
    - sg (SkipGram): Object SkipGram (from code.skipgram)
    - current_word (int): Index of the word for which we (want to) predict the context
    - contexts_word (list): Contexts obtained from some function !!!
    - negative_samples (np.array): ...
    - w0 (np.ndarray): Input weights
    - w1 (np.ndarray): Hidden weights (final embeddings)
    - epoch_error (float): Mean batch error accross current epoch
    - sgd_count (int): Number of batches already pass through SkipGram
    Output:
    - w0, w1, epoch_error, sgd_count: updated version of the Parameters.
    """
    # PREPARE DATA
    batch_contexts = contexts_word + list(negative_samples)
    target = np.array([1 for ctxt in contexts_word] + [0 for i in negative_samples])

    # FORWARD PASS
    ## Get embeddings and weights
    current_word = w0[current,:]
    context_words = w1[:, batch_contexts]

    ## Probabilities
    output = expit(np.dot(current_word, context_words))

    # BACKWARD PASS
    ## Gradient computations
    gradient_loss = gradients_custom(target, output, current_word, context_words)

    ## Update
    w0[current,:] -= sg.stepsize * gradient_loss[0]
    w1[:, batch_contexts] -= sg.stepsize * gradient_loss[1]

    # Keep track of the error (mean probabilities error)
    epoch_error += np.sum(np.abs(output - target))
    sgd_count += target.shape[0]

    return(w0, w1, epoch_error, sgd_count)

def gradients(y, ypred, W, C):
    """
    Compute the gradients of the loss described in ...
    Parameters:
    - y (np.ndarray):
    - ypred (np.ndarray): Prediction of the current words
    - W (np.ndarray):
    - C (np.ndarray):
    Output:
    - first (np.ndarray):
    - second (np.ndarray):
    """
    # Function that computes the gradients.
    root = -y*ypred*np.exp( -W.dot(C) ) + (1 - y)*(1-ypred)*np.exp( W.dot(C) )

    first = root.reshape(-1 , 1).dot(C.reshape(1 , -1))
    second = W.T.dot(root)

    return first , second

def gradients_custom(target, prediction, current_word, context_words):
    """
    Compute the gradients of the Negative sampling loss described in report.pdf
    Parameters:
    - target (np.ndarray):
    - prediction (np.ndarray): Prediction of the current words
    - current_word (np.ndarray): w0 of the current word
    - context_words (np.ndarray): w1 for context words
    Output:
    - grad_w0 (np.ndarray): gradient for w0
    - grad_w1 (np.ndarray): gradient for w1
    """
    # Compute error
    error = prediction - target

    # Exact gradients
    grad_w0 = np.sum(error*context_words, axis = 1)
    grad_w1 = np.outer(current_word, error)

    return(grad_w0, grad_w1)
