from code.utils import display_progression, sigmoid_array, assign_w2v
from code.contexts import contexts_sentence
from code.batch import run_batch
import numpy as np
import random
import pickle

class SkipGram:
    def __init__(self, sentences, embedding_dimension=100, negative_rate=5, window_size = 5, min_count = 5, epochs = 100, stepsize = 0.01, dynamic_window = False, subsampling = 1e-5):
        """
        Attributes:
            DEFAULT
            - window_size
            - min_count
            - negative_rate
            - embedding_dimension
            - sentences
            CUSTOM
            - word2count: frequency counts of sentences
            - sorted_word_count: drop words for counts < min_count
            - word2idx: from word to its index
            - normalization_cte: sum of all counts
            - word_distribution: probabilities for each word, raised to 3/4 power
            - vocab_size: vocabulary size
        """
        # ATTRIBUTES
        ## Basic attributes - given through arguments in the command line execution
        self.window_size = window_size
        self.min_count = min_count
        self.negative_rate = negative_rate
        self.embedding_dimension = embedding_dimension
        ## Custom attributes
        self.epochs = epochs
        self.stepsize = stepsize
        self.dynamic_window = dynamic_window

        # CREATE VOCABULARY
        ## Count words and create a sorted dictionary
        unique, count = np.unique(np.array([x for sentence in sentences for x in sentence]), return_counts = True)
        idx = np.where(count > min_count)
        unique, count = unique[idx], count[idx]

        ## Word count
        self.word2count = dict(zip(unique, count))
        self.word2count = dict(sorted(self.word2count.items(), key=lambda kv: kv[1]))

        ## Retrieve vocabulary size after min count
        self.vocab_size = len(self.word2count)
        self.total_words = sum(self.word2count.values())

        ## Create mapping between words and index
        self.idx2word = dict(zip(np.arange(0, self.vocab_size), self.word2count.keys()))
        self.word2idx = {v:k for k, v in self.idx2word.items()}

        # ENCODE CORPUS
        if subsampling is not None:
            # SUBSAMPLING
            frequencies = {word: count/self.total_words for word, count in self.word2count.items()}
            drop_probability = {word: (frequencies[word] - subsampling)/frequencies[word] - np.sqrt(subsampling/frequencies[word]) for word in self.word2count.keys()}
            self.train_words = {k:v for k, v in drop_probability.items() if (1 - v) > random.random()}
            self.corpus_idx = [[self.word2idx[w] for w in sent if w in self.train_words.keys()] for sent in sentences]
        else:
            self.corpus_idx = [[assign_w2v(self.word2idx, w) for w in sent] for sent in sentences]
            self.corpus_idx = [[idx for idx in sent if idx is not None] for sent in self.corpus_idx]

        # NEGATIVE SAMPLING
        self.normalization_cte = sum([occurence**0.75 for occurence in self.word2count.values()])
        self.word_distribution = {k:(v**0.75)/self.normalization_cte for k, v in self.word2count.items()}

    def train(self, frame = 100000, batch_size = 128):

        # CONTEXT CREATION
        contexts_corpus = list(map(lambda x: contexts_sentence(x, self.window_size, self.dynamic_window), self.corpus_idx))
        contexts_corpus = [item for sublist in contexts_corpus for item in sublist]
        current_words = [item for sublist in self.corpus_idx for item in sublist]

        # MODEL TRAINING
        ## Initialize uniform weights
        w0 = np.random.uniform(-1, 1, size = (self.vocab_size, self.embedding_dimension))
        w1 = np.random.uniform(-1, 1, size = (self.embedding_dimension, self.vocab_size))

        ## Negative samples
        negative_samples = np.random.choice(list(self.word2idx.values()),
                                            size = (len(contexts_corpus), self.negative_rate * self.window_size),
                                            p=list(self.word_distribution.values()),
                                            replace = True)

        ## Repeat training on whole corpus self.epochs times
        for epoch in range(0, self.epochs):
            ## Error trackings
            epoch_error = 0
            sgd_count = 1

            ## For every word in the corpus
            for i, current in enumerate(current_words):
                ## Run batch
                w0, w1, epoch_error, sgd_count = run_batch(self, current, contexts_corpus[i], negative_samples[i,:], w0, w1, epoch_error, sgd_count)
                display_progression(self, epoch, i, epoch_error, sgd_count, frame = frame, nb = len(current_words))

            self.w0 = w0
            self.w1 = w1

        return(bce_loss_history)

    def save(self, path):
        if '.pickle' not in path:
            path = path + '.pickle'
        with open(path,'wb') as handle:  # write in the file
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def similarity(self, word1, word2):
        """
        Computes similiarity between the two words. Unknown words are mapped to one common vector.
        Param word1: first word (format:string)
        Param word2: second word (format:string)
        Peturn: a float between 0 and 1 indicating the similarity (the higher the more similar).
        """
        # cosine similarity = u*v / |u|*|v|
        # v1, v2: index of word1, word2
        v1 = self.word2idx[word1]
        v2 = self.word2idx[word2]
        return (1+np.dot(self.w1[:, v1],self.w1[:, v2])/ \
                (np.linalg.norm(self.w1[:,v1]) * np.linalg.norm(self.w1[:, v2])))/2

    @staticmethod
    def load(path):
        if '.pickle' not in path:
            path = path + '.pickle'
        with open(path,'rb') as handle: # read the file
            sg = pickle.load(handle)
        return sg
