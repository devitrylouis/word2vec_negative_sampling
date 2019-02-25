"""
TODO:
- Smart vectorization for contexts_sentence
"""

import numpy as np

def contexts_sentence(encoded_sentence, window_size, dynamic_window = False):
    """
    Retrieve all contexts within a encoded_sentence.
    Parameters:
    - encoded_sentence (list of int indices):
    - window_size (list of int indices):
    - dynamic_window (bool): Whether a dynamic window should be used
    Output:
    - results (list of )
    """
    results = []
    for i, x in enumerate(encoded_sentence):
        results.append(retrieve_context(i, window_size, encoded_sentence, dynamic_window))
    return(results)

def retrieve_context(w, window_size, encoded_sentence, dynamic_window = False):
    """
    Compute the context words of a given word `w` in a `encoded_sentence`.
    Parameters:
    - encoded_sentence (list of int indices): Word indices
    - window_size (list of int indices): half size of the window
    - dynamic_window (bool): Whether a dynamic window should be used
    Output:
    - context (list): indices of context words of `w`
    """
    # Change to dynamic window
    if dynamic_window: window_size = np.random.randint(1, window_size+1)

    # List comprehension to retrieve the words
    context = encoded_sentence[max(w - window_size, 0) : w] + \
              encoded_sentence[w + 1: min(w + window_size + 1, len(encoded_sentence))]

    return(context)
