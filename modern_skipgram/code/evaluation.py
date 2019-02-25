from code.skipgram import SkipGram
from code.utils import get_test_set
from sklearn.decomposition import PCA
import numpy as np
import copy

def evaluate_w2v(w2v, sentences):
    """
    Parameters:
    - w2v (SkipGram): Pre-trained SkipGram
    - sentences (list): Clean evaluation data
    Output:
    - errors (np.array): MAE between predicted similarity and ground-truth
    """
    # Get predictions and labels
    predictions = np.array([simil(w2v, sent) for sent in sentences])
    labels = np.array([float(sent[2]) for sent in sentences])

    # Remove None
    idx = np.where(predictions != None)

    # Predictions and labels
    predictions = predictions[idx]
    labels = labels[idx]

    # Compute mean error
    errors = np.abs(predictions*10 - labels)

    return(errors)

def pca_w2v(w2v, n_components):
    """
    Apply PCA on the embeddings on a SkipGram object.
    Parameters:
    - w2v (SkipGram): Pre-trained SkipGram
    - n_components (int): Number of components to keep
    Output:
    - new_w2v (SkipGram)
    """
    new_w2v = copy.deepcopy(w2v)
    pca = PCA(n_components)
    new_w2v.w1 = pca.fit_transform(new_w2v.w1.T)
    new_w2v.w1 = new_w2v.w1.T
    return(new_w2v)

def most_similar(sg, word):
    """
    Make it a static method
    """
    results = {}
    for w in sg.word2idx.keys():
        results[w] = sg.similarity(word, w)

    results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    return(results)

def simil(w2v, sent):
    try:
        return(w2v.similarity(sent[0], sent[1]))
    except:
        return(None)
