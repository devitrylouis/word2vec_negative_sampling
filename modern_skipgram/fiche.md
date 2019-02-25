# Skipgram

Je commence par l'image de skipgram mais j'arrive pas à trouver comment l'insérer.

## Input weight matrix `w0`

The input vector $X$ is a one hot encoded vector with dimension $V$. $W$ is the weight matrix of dimension $V*N$, with $N$ being the embedding dimention. Therefore:  $W^{T}X=h$
$$
\left[ {\begin{array}{ccccccc}
   . & .&.&w_{1,k}&.&.&. \\
   . & .&.&w_{2,k}&.&.&.\\
   . & .&.&.&.&.&.\\
   . & .&.&.&.&.&.\\
   . & .&.&w_{N,k}&.&.&.\\
  \end{array} } \right] *

  \left[ {\begin{array}{c}
     . \\
     . \\
     1 \\
     . \\
     . \\
    \end{array} } \right] =

    \left[ {\begin{array}{c}
       h_{1} \\
       . \\
       h_{i} \\
       . \\
       h_{N} \\
      \end{array} } \right]
$$

It turns out that $h$ is just the $j$-th column of $W^{T}$ or the $j$-th line of $W$. $h$ is juste copying and transposing a row of the input. Therefore, we can compute h by retrieving `h=w0[context_word,:]`.

---
## Hidden weight matrix `w1`

Through the same process, the output units are obtained by multiplying the hidden units by $W'^{T}$, $V*N$ weights matrix. $U=W'^{T} * h$

$$
\left[ {\begin{array}{ccccccc}
 . & .&.&w'_{1,k}&.&.&.\\
 . & .&.&w'_{2,k}&.&.&.\\
 . & .&.&.&.&.&.\\
 . & .&.&.&.&.&.\\
 . & .&.&w'_{V,k}&.&.&.\\
\end{array} } \right] *

\left[ {\begin{array}{c}
h_{1} \\
. \\
h_{i} \\
. \\
h_{N} \\
  \end{array} } \right] =

  \left[ {\begin{array}{c}
     U_{c,1} \\
     . \\
     U_{c,i} \\
     . \\
     U_{c,V} \\
    \end{array} } \right]
$$

$U_{c,i}$ is the score of the i-th output word in c-th panel. We therefore need to use the softmax function to transform all these scores into probabilities.
$$
p(w_{c,j}=w_{O,c}|w_{I})=y_{c,j}= \frac{exp(u_{c,j})}{\sum_{j'=1}^{V}exp(u_{j'})}
$$

* $w_{c,j}$ is the j-th word on the c-th panel
* $w_{O,c}$ is the actual c-th word in the output context words
* $w_{I}$ is the input word
* $y_{c,j}$ is the output of the j-th unit of the c-th panel
* $u_{c,j}$ is the input of the j-th unit on c-th panel. However, as the output layer panels share the same weights, we have : $u_{c,j}=u_{j}$ for $c= 1,2,...,C$

For Skipgram the loss function takes this form:
$$
\begin{align}
E & =-\log p(w_{O,1},w_{O,2},...,w_{O,C}|w_{I})\\
& = -\log \prod_{c=1}^{C}\frac{exp(u_{c,j_{c}^{\ast})}}{\sum_{j'=1}^{V}exp(u_{j'})}\\
& = - \sum_{c=1}^{C}u_{j_{c}^{\ast}}+C.\log \sum_{j'=1}^{V}exp(u_{j'})
\end{align}
$$

$j_{c}^{\ast}$: the index of the actual c-th context word

## Backpropagation

We first take the derivative of E regarding the input of every unit in every panel of the output layer $u_{c,j}$:
$$
\frac{\partial E}{\partial u_{c,j}}=y_{c,j}-t_{c,j}=e_{c,j}
$$
which is the prediction error on the unit. $t_{c,j}=1 \: if  \:j=j^\ast \: and \: 0 \: otherwise$.
Thus we can define a V dimensional vector EI={$EI_{1},...,EI_{V}$} as the sum of all prediction errors over all context words: $EI_{j}=\sum_{c=1}^{C}e_{c,j}$

### Hidden $\rightarrow$ output weights

The next step is to take the derivative of E regarding the hidden $\rightarrow$ output matrix W'.
$$
\frac{\partial E}{\partial w^\prime_{i,j}}=\sum_{c=1}^{C}\frac{\partial E}{\partial u_{c,j}}.\frac{\partial u_{c,j}}{\partial w^\prime_{i,j}}=EI_{j}.h_{i}
$$

The update equation for the hidden $\rightarrow$ output matrix W' appears to be:
$$w^{\prime (new)}_{i,j}=w^{\prime (old)}_{i,j}-\eta.EI_{j}.h_{i}$$

L'explication intuitive de cette formule c'st l'histoire de closer farther, à voir si on arrive à comprendre d'ici là.

This update equation needs to be applied to every elemnt of the hidden hidden $\rightarrow$ output matrix.

### Input $\rightarrow$ Hidden weights
