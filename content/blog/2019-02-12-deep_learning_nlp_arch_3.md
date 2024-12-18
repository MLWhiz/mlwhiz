---
title: "NLP  Learning Series: Part 3 - Attention, CNN and what not for Text Classification"
date:  2019-03-09
draft: false
url : blog/2019/03/09/deeplearning_architectures_text_classification/
slug: deeplearning_architectures_text_classification
Category: ai, deep learning,kaggle, NLP
Keywords:
- artificial intelligence
-  deep learning
- kaggle
-  pytorch
-  keras to pytorch
-  keras vs pytorch
-  keras is better than pytorch
-  pytorch better than keras
-  pytorch for text classification
-  kaggle text classification
-  text classification
-  NLP
-  preprocessing
-  preprocessing for NLP
-  Natural language processing
- artificial intelligence
- deep learning
Tags:
- Natural Language Processing
- Deep Learning
- Artificial Intelligence
- Kaggle
- Best Content

Categories:
- Natural Language Processing
- Deep Learning
- Awesome Guides

description: Recently, I started up with an NLP competition on Kaggle called Quora Question insincerity challenge. It is an NLP Challenge on text classification, and as the problem has become more clear after working through the competition as well as by going through the invaluable kernels put up by the kaggle experts, I thought of sharing the knowledge. In this post, I will try to take you through some basic conventional models like TFIDF, Count Vectorizer, Hashing, etc. that have been used in text classification and try to access their performance to create a baseline.
thumbnail : /images/birnn.png
image  : /images/birnn.png
toc : false
type : post
menu: nlpseries
---

This post is the third post of the NLP Text classification series. To give you a recap, I started up with an NLP text classification competition on Kaggle called Quora Question insincerity challenge. So I thought to share the knowledge via a series of blog posts on text classification. The [first post](/blog/2019/01/17/deeplearning_nlp_preprocess/) talked about the different **preprocessing techniques that work with Deep learning models** and **increasing embeddings coverage**. In the [second post](/blog/2019/02/08/deeplearning_nlp_conventional_methods/), I talked through some **basic conventional models** like TFIDF, Count Vectorizer, Hashing, etc. that have been used in text classification and tried to access their performance to create a baseline. In this post, I delve deeper into **Deep learning models and the various architectures** we could use to solve the text Classification problem. To make this post platform generic, I am going to code in both Keras and Pytorch. I will use various other models which we were not able to use in this competition like **ULMFit transfer learning** approaches in the fourth post in the series.

**As a side note**: If you want to know more about NLP, I would like to recommend this awesome [Natural Language Processing Specialization](https://imp.i384100.net/555ABL). You can start for free with the 7-day Free Trial. This course covers a wide range of tasks in Natural Language Processing from basic to advanced: sentiment analysis, summarization, dialogue state tracking, to name a few.

So let me try to go through some of the models which people are using to perform text classification and try to provide a brief intuition for them — also, some code in Keras and Pytorch. So you can try them out for yourself.

----------
## 1. TextCNN

The idea of using a CNN to classify text was first presented in the paper [Convolutional Neural Networks for Sentence Classification](https://www.aclweb.org/anthology/D14-1181) by Yoon Kim.

**Representation:** The central intuition about this idea is to **see our documents as images**. How? Let us say we have a sentence and we have maxlen = 70 and embedding size = 300. We can create a matrix of numbers with the shape 70x300 to represent this sentence. For images, we also have a matrix where individual elements are pixel values. Instead of image pixels, the input to the tasks is sentences or documents represented as a matrix. Each row of the matrix corresponds to one-word vector.

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/text_convolution.png"  height="400" width="700" ></center>
</div>


**Convolution Idea:** While for an image we move our conv filter horizontally as well as vertically, for text we fix kernel size to filter_size x embed_size, i.e. (3,300) we are just going to move vertically down for the convolution taking look at three words at once since our filter size is 3 in this case. This idea seems right since our convolution filter is not splitting word embedding. It gets to look at the full embedding of each word. Also one can think of filter sizes as unigrams, bigrams, trigrams, etc. Since we are looking at a context window of 1,2,3, and 5 words respectively.

Here is the text classification network coded in Pytorch:

```py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN_Text(nn.Module):

    def __init__(self):
        super(CNN_Text, self).__init__()
        filter_sizes = [1,2,3,5]
        num_filters = 36
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.convs1 = nn.ModuleList([nn.Conv2d(1, num_filters, (K, embed_size)) for K in filter_sizes])
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(len(Ks)*num_filters, 1)


    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logit = self.fc1(x)
        return logit
```
And for the Keras enthusiasts:

```py
# https://www.kaggle.com/yekenot/2dcnn-textclassifier
def model_cnn(embedding_matrix):
    filter_sizes = [1,2,3,5]
    num_filters = 36

    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = Reshape((maxlen, embed_size, 1))(x)

    maxpool_pool = []
    for i in range(len(filter_sizes)):
        conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], embed_size),
                                     kernel_initializer='he_normal', activation='relu')(x)
        maxpool_pool.append(MaxPool2D(pool_size=(maxlen - filter_sizes[i] + 1, 1))(conv))

    z = Concatenate(axis=1)(maxpool_pool)
    z = Flatten()(z)
    z = Dropout(0.1)(z)

    outp = Dense(1, activation="sigmoid")(z)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
```

I am a big fan of Kaggle Kernels. One could not have imagined having all that compute for free. You can find a running version of the above two code snippets in this [kaggle kernel](https://www.kaggle.com/mlwhiz/textcnn-pytorch-and-keras). Do try to experiment with it after forking and running the code. Also please upvote the kernel if you find it helpful.

The Keras model and Pytorch model performed similarly with Pytorch model beating the keras model by a small margin. The Out-Of-Fold CV F1 score for the Pytorch model came out to be 0.6609 while for Keras model the same score came out to be 0.6559. I used the same preprocessing in both the models to be better able to compare the platforms.

----------

## 2. BiDirectional RNN(LSTM/GRU):

TextCNN works well for Text Classification. It takes care of words in close range. It can see "new york" together. However, it still can't take care of all the context provided in a particular text sequence. It still does not learn the sequential structure of the data, where every word is dependent on the previous word. Or a word in the previous sentence.

RNN help us with that. *They can remember previous information using hidden states and connect it to the current task.*

Long Short Term Memory networks (LSTM) are a subclass of RNN, specialized in remembering information for an extended period. Moreover, the Bidirectional LSTM keeps the contextual information in both directions which is pretty useful in text classification task (But won't work for a time series prediction task as we don't have visibility into the future in this case).

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/birnn.png"  height="400" width="700" ></center>
</div>

For a most simplistic explanation of Bidirectional RNN, think of RNN cell as a black box taking as input a hidden state(a vector) and a word vector and giving out an output vector and the next hidden state. This box has some weights which are to be tuned using Backpropagation of the losses. Also, the same cell is applied to all the words so that the weights are shared across the words in the sentence. This phenomenon is called weight-sharing.


<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/singlernn.png"  height="30%" width="30%" ></center>
</div>

            Hidden state, Word vector ->(RNN Cell) -> Output Vector , Next Hidden state

For a sequence of length 4 like **"you will never believe"**, The RNN cell gives 4 output vectors, which can be concatenated and then used as part of a dense feedforward architecture.

In the Bidirectional RNN, the only change is that we read the text in the usual fashion as well in reverse. So we stack two RNNs in parallel, and hence we get 8 output vectors to append.

Once we get the output vectors, we send them through a series of dense layers and finally a softmax layer to build a text classifier.

In most cases, you need to understand how to stack some layers in a neural network to get the best results. We can try out multiple bidirectional GRU/LSTM layers in the network if it performs better.

Due to the limitations of RNNs like not remembering long term dependencies, in practice, we almost always use LSTM/GRU to model long term dependencies. In such a case you can think of the RNN cell being replaced by an LSTM cell or a GRU cell in the above figure. An example model is provided below. You can use CuDNNGRU interchangeably with CuDNNLSTM when you build models. (CuDNNGRU/LSTM are just implementations of LSTM/GRU that are created to run faster on GPUs. In most cases always use them instead of the vanilla LSTM/GRU implementations)

So here is some code in Pytorch for this network.

```py
class BiLSTM(nn.Module):

    def __init__(self):
        super(BiLSTM, self).__init__()
        self.hidden_size = 64
        drp = 0.1
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(self.hidden_size*4 , 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drp)
        self.out = nn.Linear(64, 1)


    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))

        h_lstm, _ = self.lstm(h_embedding)
        avg_pool = torch.mean(h_lstm, 1)
        max_pool, _ = torch.max(h_lstm, 1)
        conc = torch.cat(( avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)
        return out
```

Also, here is the same code in Keras.

```py
# BiDirectional LSTM
def model_lstm_du(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    '''
    Here 64 is the size(dim) of the hidden state vector as well as the output vector. Keeping return_sequence we want the output for the entire sequence. So what is the dimension of output for this layer?
        64*70(maxlen)*2(bidirection concat)
    CuDNNLSTM is fast implementation of LSTM layer in Keras which only runs on GPU
    '''
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    conc = Dense(64, activation="relu")(conc)
    conc = Dropout(0.1)(conc)
    outp = Dense(1, activation="sigmoid")(conc)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
```


You can run this code in my [BiLSTM with Pytorch and Keras kaggle kernel](https://www.kaggle.com/mlwhiz/bilstm-pytorch-and-keras) for this competition. Please do upvote the kernel if you find it helpful.

In the BiLSTM case also, Pytorch model beats the keras model by a small margin. The Out-Of-Fold CV F1 score for the Pytorch model came out to be 0.6741 while for Keras model the same score came out to be 0.6727. This score is around a 1-2% increase from the TextCNN performance which is pretty good. Also, note that it is around 6-7% better than conventional methods.

--------

## 3. Attention Models

Dzmitry Bahdanau et al first presented attention in their paper [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) but I find that the paper on [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf) written jointly by CMU and Microsoft in 2016 is a much easier read and provides more intuition.

So let us talk about the intuition first. In the past conventional methods like TFIDF/CountVectorizer etc. we used to find features from the text by doing a keyword extraction. Some word is more helpful in determining the category of a text than others. However, in this method we sort of lost the sequential structure of the text. With LSTM and deep learning methods, while we can take care of the sequence structure, we lose the ability to give higher weight to more important words.
**Can we have the best of both worlds?**

The answer is Yes. Actually, **Attention is all you need**. In the author's words:

> Not all words contribute equally to the representation of the sentence meaning. Hence, we introduce attention mechanism to extract such words that are important to the meaning of the sentence and aggregate the representation of those informative words to form a sentence vector

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/birnn attention.png"  height="400" width="700" ></center>
</div>

In essence, we want to create scores for every word in the text, which is the attention similarity score for a word.

To do this, we start with a weight matrix(W), a bias vector(b) and a context vector u. The optimization algorithm learns all of these weights. On this note I would like to highlight something I like a lot about neural networks - If you don't know some params, let the network learn them. We only have to worry about creating architectures and params to tune.

Then there are a series of mathematical operations. See the figure for more clarification. We can think of u1 as nonlinearity on RNN word output. After that v1 is a dot product of u1 with a context vector u raised to exponentiation. From an intuition viewpoint, the value of v1 will be high if u and u1 are similar. Since we want the sum of scores to be 1, we divide v by the sum of v’s to get the Final Scores,s

These final scores are then multiplied by RNN output for words to weight them according to their importance. After which the outputs are summed and sent through dense layers and softmax for the task of text classification.

Here is the code in Pytorch. **Do try to read through the pytorch code for attention layer.** It just does what I have explained above.

```py
class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)

class Attention_Net(nn.Module):
    def __init__(self):
        super(Attention_Net, self).__init__()
        drp = 0.1
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.embedding_dropout = nn.Dropout2d(0.1)
        self.lstm = nn.LSTM(embed_size, 128, bidirectional=True, batch_first=True)
        self.lstm2 = nn.GRU(128*2, 64, bidirectional=True, batch_first=True)

        self.attention_layer = Attention(128, maxlen)

        self.linear = nn.Linear(64*2 , 64)
        self.relu = nn.ReLU()
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))
        h_lstm, _ = self.lstm(h_embedding)
        h_lstm, _ = self.lstm2(h_lstm)
        h_lstm_atten = self.attention_layer(h_lstm)
        conc = self.relu(self.linear(h_lstm_atten))
        out = self.out(conc)
        return out
```

Same code for Keras.

```py
def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


def model_lstm_atten(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    x = AttentionWithContext()(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

```

Again, my [Attention with Pytorch and Keras Kaggle kernel](https://www.kaggle.com/mlwhiz/attention-pytorch-and-keras) contains the working versions for this code. Please do upvote the kernel if you find it useful.

This method performed well with Pytorch CV scores reaching around 0.6758 and Keras CV scores reaching around 0.678. **This score is more than what we were able to achieve with BiLSTM and TextCNN.** However, please note that we didn't work on tuning any of the given methods yet and so the scores might be different.

With this, I leave you to experiment with new architectures and playing around with stacking multiple GRU/LSTM layers to improve your network performance. You can also look at including more techniques in these network like Bucketing, handmade features, etc. Some of the tips and new techniques are mentioned here on my blog post: [What my first Silver Medal taught me about Text Classification and Kaggle in general?](/blog/2019/02/19/siver_medal_kaggle_learnings/). Also, here is another Kaggle kernel which is [my silver-winning entry](https://www.kaggle.com/mlwhiz/multimodel-ensemble-clean-kernel?scriptVersionId=10279838) for this competition.

----------

## Results

Here are the final results of all the different approaches I have tried on the Kaggle Dataset. I ran a 5 fold Stratified CV.

### a. Conventional Methods:

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/results_conv.png"  style="height:40%;width:40%"></center>
</div>

### b. Deep Learning Methods:

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/results_deep_learning.png"  style="height:50%;width:50%"></center>
</div>


**PS:** Note that I didn't work on tuning the above models, so these results are only cursory. You can try to squeeze more performance by performing hyperparams tuning [using hyperopt](/blog/2017/12/28/hyperopt_tuning_ml_model/) or just old fashioned Grid-search.

----------
## Conclusion

In this post, I went through with the explanations of various deep learning architectures people are using for Text classification tasks. In the next post, we will delve further into the next new phenomenon in NLP space - Transfer Learning with BERT and ULMFit. Follow me up at [Medium](https://mlwhiz.medium.com/) or Subscribe to my blog to be informed about my next post.

If you want to know more about NLP, I would like to recommend this awesome [Natural Language Processing Specialization](https://imp.i384100.net/555ABL). You can start for free with the 7-day Free Trial. This course covers a wide range of tasks in Natural Language Processing from basic to advanced: sentiment analysis, summarization, dialogue state tracking, to name a few.

Let me know if you think I can add something more to the post; I will try to incorporate it.

Cheers!!!
