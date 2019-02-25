from IPython.display import clear_output

import numpy as np
import pandas as pd
import string

def text2sentences(path):
    """
    Read the data and apply early pre-processing (see preprocess_line)
    Parameter:
    - path (str): relative path of the training data
    Output:
    - sentences (list): Pre-processed sentences of the corpus
    """
    # Allocate results
    sentences = []
    with open(path) as f:
        # Reading data
        for l in f:
            # Preprocessing
            sentences.append(preprocess_line(l))
    return sentences

def preprocess_line(line):
    """
    Apply a whitespace stemmer, and punctuation / digits removal.
    Parameters:
    - line (str): sentence
    Output:
    - clean_line (list): clean words
    """
    lower = line.lower()
    punctuation = ''.join(c for c in lower if c not in string.punctuation)
    digits = ''.join([i for i in punctuation if not i.isdigit()])
    clean_line = digits.split()
    return(clean_line)

def get_test_set(path):
    """
    Read SimLex 999 and return clean, nested lists.
    Parameters:
    - path (str): Path of the file
    Output:
    - sentences (list): Clean evaluation data
    """
    sentences = []
    with open(path) as f:
        for i, l in enumerate(f):
            if i > 1:
                tmp = l.split('\t')
                tmp[2] = tmp[2][0:-2]
                if tmp[2] != '':
                    sentences.append(tmp)

    return sentences

def loadPairs(path):
    data = pd.read_csv(path,delimiter='\t')
    pairs = zip(data['word1'],data['word2'],data['similarity'])
    return pairs

def display_progression(sg, epoch, s, epoch_error, sgd_count, frame = 1000, nb = 1):
    if s % frame == 0:
        clear_output()
        print("Epochs: " + str(epoch + 1))
        print('- Progression : ' + str(s) + ' contexts done out of ' + str(nb))
        print('- Mean loss so far : ' + str(epoch_error/float(sgd_count)))

def sigmoid_array(x):
    return 1 / (1 + np.exp(-x))

def assign_w2v(word2idx, word):
    """
    Try to retrieve the idx of a given word. None otherwise
    """
    try:
        return(word2idx[word])
    except:
        return(None)
