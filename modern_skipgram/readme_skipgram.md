# SkipGram rocks!

As suggested by the given template, we will construct a class `SkipGram` that encapsulates all the hard work. The OOP frameword will also prove useful in efficiently running experiments.

<b>Table of content:</b>

1. [What is skipgram?](#skipgram)
    * [What is learnt?](#embeddings)
    * [How does it learn?](#neural_net)
    * [How to learn even faster?](#negative_sampling)
    * [Draw Negative samples ](#negative_sampling)
    * [Back-propagation equations - TODO](#back_propagation)
2. [Pre-processing is done in `__init__`!](#init)
    * [Text preprocessing](#text-preprocessing)
    * [Hyperparameters](#hyperparameters)
    * [Vocabulary](#vocabulary)
    * [Corpus encoding](#encoding)
    * [Negative sample distribution](#neg_sample_dist)
3. [The full `train` procedure](#train)
    * [The gradients in python](#gradients)
    * [A naïve learning procedure](#naive)
    * [Retrieve the contexts](#contexts)
    * [How to construct and learn a batch? - TODO](#batch)
    * [Negative sampling in practice](#practical_ng)
    * [Optimizing context and learning computations](#optimizing)
4. [Improve word2vec](#improve)
    * [Dynamic window size](#dynamic_window)
    * [Subsampling and rare word pruning](#subsampling)
5. [Experiments](experiments)
    * [Evaluation with SimLex](#evaluation)
    * [Post-processing embeddings with PCA](#pca)
6. [Future work](#future)
    * [Character n-grams](#ngrams)
    * [Algebraic properties of the embeddings](#algebraic)

---

# 1. What is SkipGram? Section to include in LaTeX

> <b>TL; DR:</b> "[Consinuous SkipGram predicts] the current word based on its context, and tries to maximize classification of a word based on another word in the same sentence. More precisely, we use each current word as an input to a log-linear classifier with continuous projection layer, and predict words within a certain range before and after the current word." - [Original paper](https://arxiv.org/pdf/1301.3781.pdf)

In this section, we explain the inner-workings of the Skip Gram model in its intricate details. Specifically, we answer the questions:
- What is learnt?
- How does it learn?
- How to learn even faster?

### 1.1. What is learnt? <a class="anchor" id="embeddings"></a>

While the <u>general finality</u> of SkipGram is to <u>learn word embeddings</u> (i.e. find a latent representation of the words), its <u>explicit goal</u> remains to <u>learn how to predict the context</u> $c$ of a given word $w$.

SkipGram uses a <u>parametrized model</u> (with parameters $\theta$) whose aim is to <u>maximize the corpus probability</u>:

$$
\underset{\theta}{\text{arg max}} \prod_{w \in corpus} \Big[ \prod_{c\in C(w)}\ p(c|w; \theta) \Big]
$$

where $p(c|w; \theta)$ is the probability that the context word $c$ was near $w$.

### 1.2. How does it learn? <a class="anchor" id="neural_net"></a>

The <b>most successfull parametrized model</b> for SkipGram to date is a <b>neural network with one hidden layer</b> $h$. The neural network can adequately model the conditional probability $p(c| x: \theta)$ using the softmax:

$$
p(c \mid w; \theta) \propto e^{v_{c}\cdot v_{w}}
$$

where $v_{c}$ and $v_{w} \in \mathbb{R}^{d}$ are vector representations for $c$ and $w$ respectively, and $C$ is the set of all available contexts.

![SkipGram architecture](https://www.researchgate.net/publication/322905432/figure/fig1/AS:614314310373461@1523475353979/The-architecture-of-Skip-gram-model-20.png)

However, this approach is limited because as the size of our vocabulary increases, the normalization constant of the softmax needs to compute a growing number of terms $e^{v_{c'}\cdot v_{w}}$, $\forall c' \in \text{vocabulary}$.

### 1.3. How to learn even faster? <a class="anchor" id="negative_sampling"></a>

[Mikolov et al.](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) introduce the <u>negative sampling objective</u>, which is <b>much (much) faster</b> to optimize than the classical softmax. The trick they use is that they instead aim to maximize the probabilities that all of the observations indeed came from the data:

$$
\underset{\theta}{\text{arg max}} \prod_{(w, c) \in D} p(D = 1 | c, w; \theta) \Leftrightarrow \underset{\theta}{\text{arg max}} \sum_{(w, c) \in D} \text{log} \frac{1}{1+e^{-v_{c}\cdot v_{w}}}
$$

See [tutorial_paper](https://www.cs.bgu.ac.il/~yoavg/publications/negative-sampling.pdf) for the full derivation equivalence. However, in this setting, a <b>trivial solution</b> (see below) can be obtained and leads to a <b>non-injective representation</b> (two identical vectors for two distinct words).

> <b>Non-injective representation from [tutorial_paper](https://www.cs.bgu.ac.il/~yoavg/publications/negative-sampling.pdf):</b> <br>
> Set $\theta$ such that $p(D = 1|w, c; \theta) = 1$ for every pair $(w, c)$. This can be easily achieved by setting $\theta$ such that $v_{c} = v_{w}$ and $v_{c} · v_{w} = K$ for all $v_{c}, v_{w}$, where $K$ is large enough number (practically, we get a probability of $1$ as soon as $K \approx 40$).

To palliate this issue, negative sampling introduces a <u>generated dataset</u> $D'$ made of <u>incorrect pairs</u> to "balance" the probabilities. This result in the negative sampling loss:

$$
\underset{\theta}{\text{arg max}} \prod_{(w, c) \in D} p(D = 0 | c, w; \theta)\ \prod_{(w, c) \in D} p(D = 1 | c, w; \theta)
$$

Similarly than before, but with the [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) $\sigma$, we obtain:

$$
\underset{\theta}{\text{arg max}} \sum_{(w, c) \in D} \text{log} \  \sigma(-v_{c}\cdot v_{w}) + \sum_{(w, c) \in D'} \text{log} \ \sigma(v_{c}\cdot v_{w})
$$

To turn this into a loss, with $x = v_{w}$ and $y = v_{c}$ we simply do:

$$
L =  \sum_{(x, y) \in D} - \text{log} \  \sigma(-y\cdot x) + \sum_{(x, y) \in D'} - \text{log} \ \sigma(y\cdot x)
$$

$$
L =  \sum_{(x, y) \in D} x\cdot y - \text{log} \  \sigma(x\cdot y) + \sum_{(x, y) \in D'} - \text{log} \ \sigma(y\cdot x)
$$

$$
L =  \sum_{(x, y) \in D} x\cdot y - \text{log} \  \sigma(x\cdot y) + \sum_{(x, y) \in D'} - \text{log} \ \sigma(y\cdot x)
$$

$$
L =  \sum_{(x, y) \in D} x\cdot y \ - \sum_{(x, y) \in D\cup D'} \text{log} \ \sigma(y\cdot x)
$$

### 1.4. Draw Negative samples in practice

For $k$ negative samples, the generated dataset $D'$ is $k$ times larger than $D$, and for each $(w, c) \in D$ we construct $k$ samples $(w, c_{1}), . . . ,(w, c_{k})$, where each $c_{j}$ is drawn according to its <u>unigram distribution</u> raised to the $3/4$ power. This is equivalent to drawing the samples $(w, c)$ in $D'$ from the distribution

$$(w, c) \sim p_{words}(w) p_{contexts}(c) 3/4 Z$$

where $p_{text{words}}(w)$ and $p_{text{contexts}}(c)$ are the unigram distributions of words and contexts respectively, and $Z$ is a normalization constant.

The authors state in their paper that they tried a number of variations on this equation, and the one which performed best was to raise the word counts to the 3/4 power:

has the tendency to increase the probability for less frequent words and decrease the probability for more frequent words.

In the work of Mikolov et al. each context is a word (and all words appear as contexts), and so $p_{text{context}}(x) = p_{words}(x) = \frac{count(x)}{|\text{vocab size}|}$.

### 1.5. Backpropagation equations  <a class="anchor" id="back_propagation"></a>

In this section, we derive the equations of backpropagation with negative sampling.

The derivatives are:

SIMO

- [Word2vec and backpropagation](http://www.claudiobellei.com/2018/01/06/backprop-word2vec/)
- [Word2vec and backpropagation 2](https://arxiv.org/pdf/1411.2738.pdf?fbclid=IwAR1CsTlSXrZYhCeh6nM8ohUSSh2tWRgBEjkgdCAvRKXVtB16sFbt5XPeIis)

---

## 2. Preprocessing is done with `__init__` method <a class="anchor" id="init"></a>

As the resulting `SkipGram` will be trained on a <u>unique corpus</u>, many tasks specific to the corpus will be done once (*) within our `SkipGram`. In consequence, our `__init__` method  will serve four goals:

### 2.1. Text preprocessing <a class="anchor" id="text-preprocessing"></a>

We use a white space stemmer as a baseline in this work. Additionally, we remove punctuations as it oftentimes leads to duplicate of words ('bye' and 'bye.'). We choose to use list comprehension motivated by this [Quora question](https://www.quora.com/How-do-I-remove-punctuation-from-a-Python-string). This can be optimized using [C based look-up tables](https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python). We also [remove numbers](https://stackoverflow.com/questions/12851791/removing-numbers-from-string).

```python
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
```

### 2.2. Hyperparameters <a class="anchor" id="hyperparameters"></a>

We <u>set the hyperparameters</u> specific to the model (`window_size`, `embedding_dimension`...) as attributes and we add all remaining hyperparameters (`epochs`, `stepsize`...) for to centralize them altogether.

```python
# ATTRIBUTES
## Basic attributes
self.window_size = window_size
self.min_count = min_count
self.negative_rate = negative_rate
self.embedding_dimension = embedding_dimension
## Custom attributes
self.epochs = epochs
self.stepsize = stepsize
self.dynamic_window = dynamic_window
self.subsampling = None or 1e-5
```

<i>Edit:</i> We added several attributes corresponding to the use of techniques such as subsampling:
- `dynamic_window` as an attribute of the model.
- `subsampling` set to either the threshold or to `None`, if subsampling should not be applied.

### 2.3. Vocabulary <a class="anchor" id="vocabulary"></a>

The vocabulary is a key component of the SkipGram model. Its main goal is to construct an integer indexing of the words (e.g. two dictionaries `self.word2idx` and `self.idx2word`). For convenience purposes, the indexes are sorted by the number of occurences. We use `np.unique` with `return_counts = True` to obtain the `unique` words of the corpus and their respective `count`. We save the number of occurences in a dictionary `self.wordcount` which will be useful for subsampling.

```python
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

## Create mapping between words and index
self.idx2word = dict(zip(np.arange(0, self.vocab_size), self.word2count.keys()))
self.word2idx = {v:k for k, v in self.idx2word.items()}
```

### 2.4. Corpus encoding <a class="anchor" id="encoding"></a>

We convert the list of sentences by replacing each word by its index from `word2idx`. List comprehension leads to more efficient computations of words contexts and is lighter in terms of storage.

```python
# Encode corpus
self.corpus_idx = [[self.word2idx[w] for w in sent] for sent in sentences]
```

### 2.5. Negative sampling distribution <a class="anchor" id="neg_sample_dist"></a>

Prepare Negative sampling upstram (because *) by computing its word distribution `word_distribution` and the corresponding `normalization_cte` (see Section 3 for more details). Additionally, as these solely rely on the occurences of each word (see `word2count`), it makes sense to integrate it in the `__init__`.

```python
# Create distribution
self.normalization_cte = sum([occurence**0.75 for occurence in self.word2count.values()])
self.word_distribution = {k:v**0.75/self.normalization_cte for k, v in self.word2count.items()}
```

This distribution is only sampled during the training phase, in `run_batch`!

## 3. `train` is all about learning in/out contexts <a class="anchor" id="train"></a>

### 3.0. The gradients in python <a class="anchor" id="gradients"></a>

Learning is done with gradient descent (see `report.pdf` (section I) for the mathematical derivation). I python, we can vectorize the gradient computations with:

```python
def gradients_custom(target, prediction, current_word, context_words):
    # Compute error
    error = prediction - target

    # Exact gradients
    grad_w0 = np.sum(error*context_words, axis = 1)
    grad_w1 = np.outer(current_word, error)

    return(grad_w0, grad_w1)
```

### 3.1. An optimized learning procedure <a class="anchor" id="naive"></a>

The `train` method incorporates the three tasks of SkipGram coupled with Negative sampling:

1. Compute the <u>in-context</u> with sliding windows around the centered word `w`
1. Compute the <u>out-contexts</u> with negative sampling
1. <u>Run batch</u> with target word (label $1$) and its negative samples (labels $0$)

Let's initialize uniformly the weights:

```python
w0 = np.random.uniform(-1, 1, size = (self.vocab_size, self.embedding_dimension))
w1 = np.random.uniform(-1, 1, size = (self.embedding_dimension, self.vocab_size))
```

At first, and because all these tasks require going over all the words of the corpus once, we used nested `for` loops, we wrote something along the lines of:

```python
# Each epochs go over all sentences of the corpus.
for epoch in range(0, self.epochs):
    # The current sentence of the corpus.
    for s, encoded_sentence in enumerate(self.corpus_idx):
        # Retrieve the `context` words for each `w`
        for w, token in enumerate(encoded_sentence):
            # For each context word and its drawn negative samples
            context = retrieve_context(w, self.window_size, encoded_sentence)
            for context_word in context:
                # Batch backprop of the neural network
                w0, w1, ... = run_batch(w0, w1, ...)
```

This approach proved to be highly inefficient because except for running the batch in itself, the other tasks like context retrieval and drawing negative samples (see section 3.4.) can be done `ONCE` efficiently. For the contexts, we use python `map` operator instead of `for` loop.

```python
# Context creation
contexts_corpus = map(lambda x: contexts_sentence(x, self.window_size, self.dynamic_window), self.corpus_idx)
# Flatten nested list
contexts_corpus = [item for sublist in contexts_corpus for item in sublist]
```

Altogether we obtain the <b>optimized learning procedure</b>:

```python
## Repeat training on whole corpus self.epochs times
for epoch in range(0, self.epochs):
    ## For every word in the corpus
    for i, current in enumerate(current_words):
        w0, w1, ... = run_batch(self, current, contexts_corpus[i], w0, w1, ...)
```

This version is inefficient for many reasons (see 3.4.) but allowed us to make `SkipGram` work in the first place.

### 3.2. Retrieve the contexts <a class="anchor" id="contexts"></a>

The basic idea behind computing `context_word` is to go over all the words in the corpus (much like one would read a corpus) and see which word is near the current word within a given window. At a given word in this "reading", we retrieve its nearby words in python using:

```python
def retrieve_context(w, window_size, encoded_sentence, dynamic_window = False):

    # Change to dynamic window
    if dynamic_window: window_size = np.random.randint(1, window_size+1)

    # List comprehension to retrieve the words
    context = encoded_sentence[max(w - window_size, 0) : w] + \
              encoded_sentence[w + 1: min(w + window_size + 1, len(encoded_sentence))]

    return(context)
```

### 3.3. How to construct and learn a batch? - Stucture OK <a class="anchor" id="batch"></a>

While the the weights `w0` and `w1` are initialized prior the epochs loop, the whole embedding learning can be boiled down to a <u>batch</u>. Word2vec thereby learns by running (lots of) batches, with each batch consisting of the `current word`, `contexts_word` and corresponding `negative_samples`.

```python
def run_batch(sg, current, contexts_word, negative_samples, w0, w1, epoch_error, sgd_count):
    # PREPARE BATCHE
    # FORWARD PASS
    # BACKWARD PASS
    return(w0, w1, epoch_error, sgd_count)
```
<b>Preparing the batch:</b> All we need are the in and out contexts (see `negative_samples` and `contexts_word`), because we know their respective `labels`. The `negative_samples` are drawn at each epoch (see 3.4.) and passed through
```python
batch_contexts = contexts_word + list(negative_samples)
target = np.array([1 for ctxt in contexts_word] + [0 for i in negative_samples])
```

<b>Forward pass:</b> We retrieve the `w0` of the current word and the embeddings `w1` of the context words. We take the sigmoid of their dot product.
```python
# Get embeddings and weights
current_word = w0[current,:]
context_words = w1[:, batch_contexts]
# Probabilities
output = expit(np.dot(current_word, context_words))
```

<b>Backward pass:</b> We compute the gradients as explained in section 3.0. and update the weights.
```python
# Gradient computations
gradient_loss = gradients_custom(target, output, current_word, context_words)
# Update
w0[current,:] -= sg.stepsize * gradient_loss[0]
w1[:, batch_contexts] -= sg.stepsize * gradient_loss[1]
```

<i>Implementation note:</i> We track the progress with `epoch_error` and `sgd_count`.

```python
# Keep track of the error (mean probabilities error)
epoch_error += np.sum(np.abs(output - target))
sgd_count += target.shape[0]
```

### 3.4. Negative sampling in practice<a class="anchor" id="practical_ng"></a>


The Negative sampling works by generating $k$ negative samples for each `context_word`. The definition of the distribution used for sampling has been done in section 2.5. As for sampling, it is standard to proceed with `np.random.choice`. At first, they were drawn within each batch. This slow down the learning process immensely. Therefore, we draw all negative samples at once before the first epoch (with `replace = True`). In python, this translates to:

```python
negative_samples = np.random.choice(list(self.word2idx.values()),
                                    size = (len(contexts_corpus), self.negative_rate * self.window_size),
                                    p=list(self.word_distribution.values()),
                                    replace = True)
```



## 4. Improve Word2Vec <a class="anchor" id="improve"></a>

### 4.1. Dynamic window size <a class="dynamic_window" id="embeddings"></a>

As a dynamic window size [arguably leads](https://www.cs.bgu.ac.il/~yoavg/publications/negative-sampling.pdf) to better results, we integrate this to our code using the attribute `self.dynamic_window = True` to the `__init__` method. As for the actual computations, we use [randint](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randint.html) function from numpy. Altogether, this gives

```python
if self.dynamic_window: win = np.random.randint(1, self.window_size + 1)
else: win = self.window_size
```

### 4.2. Effect of subsampling and rare-word pruning <a class="anchor" id="subsampling"></a>

<b>Min count:</b> Trying to learn contexts of rare words in the corpus is problematic for SkipGram, as there is not enough samples to train properly. And because there are less contexts to learn, this boosts the overall training speed.
We retain words with at least `self.min_count = 5` occurences.

<b>Subsampling:</b> Implemented as is, the frequent words such as ‘the’, ‘is’ undermines the ability of SkipGram to learn great embeddings. To tackle this issue, frequent words are down-sampled. The motivation behind this variation is that frequently occuring words hold less discriminative power. The underlying motivation is the effect of increasing both the effective window size and its quality for certain words. According to Mikolov et al. (2013), sub-sampling of frequent words improves the quality of the resulting embedding on some benchmarks.

For a given threshold $t$, we remove a word with frequency $f$ with probability:

$$
p = \frac{f-t}{f}-\sqrt{\frac{t}{f}}
$$

```python
frequencies = {word: count/self.total_words for word, count in self.word2count.items()}
drop_probability = {word: (frequencies[word] - subsampling)/frequencies[word] - np.sqrt(subsampling/frequencies[word]) for word in self.word2count.keys()}
self.train_words = {k:v for k, v in drop_probability.items() if (1 - v) > random.random()}
self.corpus_idx = [[self.word2idx[w] for w in sent if w in self.train_words.keys()] for sent in sentences]
```

<i>Implementation note:</i> Importantly, these words are removed from the text before generating the contexts (in `__init__`).

## 5. Experiments <a class="anchor" id="experiments"></a>

### 5.1. Evaluation <a class="anchor" id="evaluation"></a>

[Example of evaluation](https://github.com/mfaruqui/eval-word-vectors/blob/master/data/word-sim/EN-SIMLEX-999.txt)

### 5.2. Post-processing embeddings <a class="anchor" id="pca"></a>

[Post-processing = PCA](https://arxiv.org/pdf/1702.01417.pdf)

## 6. Future work <a class="anchor" id="future"></a>

### 6.1. Character n-grams <a class="anchor" id="ngrams"></a>

This is motivated by [these answers](https://www.quora.com/What-is-the-main-difference-between-word2vec-and-fastText). See [python implementation](https://stackoverflow.com/questions/18658106/quick-implementation-of-character-n-grams-using-python). This [one](http://locallyoptimal.com/blog/2013/01/20/elegant-n-gram-generation-in-python/) seems more nice and efficient though.

### 6.2. Algebraic properties of the embeddings <a class="anchor" id="algebraic"></a>

## 7. Sources

- [README example of w2v](http://www.telesens.co/2018/02/08/word2vec-experiments/#3_Text_Sub-Sampling_Techniques)
