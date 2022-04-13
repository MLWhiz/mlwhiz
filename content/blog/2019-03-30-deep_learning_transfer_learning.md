---
title: "NLP  Learning Series: Part 4 - Transfer Learning Intuition for Text Classification"
date:  2019-03-30
draft: false
url : blog/2019/03/30/transfer_learning_text_classification/
slug: transfer_learning_text_classification
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

Categories:
- Natural Language Processing
- Deep Learning
- Awesome Guides

Tags:
- Natural Language Processing
- Deep Learning
- Artificial Intelligence
- Kaggle

description: Recently, I started up with an NLP competition on Kaggle called Quora Question insincerity challenge. It is an NLP Challenge on text classification, and as the problem has become more clear after working through the competition as well as by going through the invaluable kernels put up by the kaggle experts, I thought of sharing the knowledge. In this post, I will try to take you through some basic conventional models like TFIDF, Count Vectorizer, Hashing, etc. that have been used in text classification and try to access their performance to create a baseline.
thumbnail : /images/nlp_tl/spiderman.jpeg
image :  /images/nlp_tl/spiderman.jpeg
toc : false
type : post
---


This post is the fourth post of the NLP Text classification series. To give you a recap, I started up with an NLP text classification competition on Kaggle called Quora Question insincerity challenge. So I thought to share the knowledge via a series of blog posts on text classification. The [first post](/blog/2019/01/17/deeplearning_nlp_preprocess/) talked about the different **preprocessing techniques that work with Deep learning models** and **increasing embeddings coverage**. In the [second post](/blog/2019/02/08/deeplearning_nlp_conventional_methods/), I talked through some **basic conventional models** like TFIDF, Count Vectorizer, Hashing, etc. that have been used in text classification and tried to access their performance to create a baseline. In the [third post](/blog/2019/03/09/deeplearning_architectures_text_classification/), I delved deeper into **Deep learning models and the various architectures** we could use to solve the text Classification problem. In this post, I will try to use ULMFit model which is a transfer learning approach to this data.

**As a side note**: If you want to know more about NLP, I would like to recommend this awesome [Natural Language Processing Specialization](https://coursera.pxf.io/9WjZo0). You can start for free with the 7-day Free Trial. This course covers a wide range of tasks in Natural Language Processing from basic to advanced: sentiment analysis, summarization, dialogue state tracking, to name a few.

Before introducing the notion of transfer learning to NLP applications, we will first need to understand a little bit about Language models.

---

## Language Models And NLP Transfer Learning Intuition:

In very basic terms the objective of the language model is to **predict the next word given a stream of input words.** In the past, many different approaches have been used to solve this particular problem. Probabilistic models using Markov assumption is one example of this sort of models.

$$ P(W\_n) = P(W\_n|W\_{n-1}) $$

In the recent era, people have been using *RNNs/LSTMs* to create such language models. They take as input a word embedding and at each time state return the probability distribution of next word probability over the dictionary words. An example of this is shown below in which the below Neural Network uses multiple stacked layers of RNN cells to learn a language model to predict the next word.

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/nlp_tl/language_model.png"  height="400" width="700" ></center>
</div>

*Now why do we need the concept of Language Modeling? Or How does predicting the next word tie with the current task of text classification?* The intuition ties to the way that the neural network gets trained. The neural network that can predict the next word after being trained on a massive corpus like Wikipedia already has learned a lot of structure in a particular language. Can we use this knowledge in the weights of the network for our advantage? Yes, we can, and that is where the idea of Transfer Learning in NLP stems from. So to make this intuition more concrete, Let us think that our neural network is divided into two parts -

- **Language Specific**: The lower part of the neural network is language specific. That is it learns the features of the language. This part could be used to transfer our knowledge from a language corpus to our current task
- **Task Specific**: I will call the upper part of our network as task specific. The weights in these layers are trained so that it learns to predict the next word.

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/nlp_tl/language_model_2.png"  height="400" width="700" ></center>
</div>

Now as it goes in a lot of transfer learning models for Image, we stack the Language Specific part with some dense and softmax layers(Our new task) and train on our new task to achieve what we want to do.

---

## ULMFit:

Now the concept of Transfer learning in NLP is not entirely new and people already used Language models for transfer learning back in 2015-16 without good result. So what has changed now?

The thing that has changed is that people like Jeremy Howard and Sebastian Ruder have done a lot of research on how to train these networks. And so we have achieved state of the art results on many text datasets with Transfer Learning approaches.

Let's follow up with the key research findings in the [ULMFit paper](https://arxiv.org/pdf/1801.06146.pdf) written by them along with the code.

---

## Change in the way Transfer Learning networks are trained:

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/nlp_tl/ulmfit_training.png"  height="400" width="700" ></center>
</div>

Training a model as per ULMFiT we need to take these three steps:

a) **Create a Base Language Model:** Training the language model on a general-domain corpus that captures high-level natural language features
b) **Finetune Base Language Model on Task Specific Data:** Fine-tuning the pre-trained language model on target task data
c) **Finetune Base Language Model Layers + Task Specific Layers on Task Specific Data:** Fine-tuning the classifier on target task data

So let us go through these three steps one by one along with the code that is provided to us with the FastAI library.

#### a) Create a Base Language Model:

This task might be the most time-consuming task. This model is analogous to resnet50 or Inception for the vision task. In the paper, they use the language model AWD-LSTM, a regular LSTM architecture trained with various tuned dropout hyperparameters. This model was trained on Wikitext-103 consisting of 28,595 preprocessed Wikipedia articles and 103 million words. We won't perform this task ourselves and will use the fabulous FastAI library to use this model as below. The code below will take our data and preprocess it for usage in the AWD_LSTM model as well as load the model.

```py
# Language model data : We use test_df as validation for language model
data_lm = TextLMDataBunch.from_df(path = "",train_df= train_df ,valid_df = test_df)
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)
```

It is also where we preprocess the data as per the required usage for the FastAI models. For example:

```py
print(train_df)
```

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/nlp_tl/train_df.png"  height="400" width="700" ></center>
</div>


```py
print(data_lm)
```

```
TextLMDataBunch;

Train: LabelList (1306122 items)
x: LMTextList
xxbos xxmaj how did xxmaj quebec nationalists see their province as a nation in the 1960s ?,xxbos xxmaj do you have an adopted dog , how would you encourage people to adopt and not shop ?,xxbos xxmaj why does velocity affect time ? xxmaj does velocity affect space geometry ?,xxbos xxmaj how did xxmaj otto von xxmaj guericke used the xxmaj magdeburg hemispheres ?,xxbos xxmaj can i convert montra xxunk d to a mountain bike by just changing the tyres ?
y: LMLabelList
,,,,
Path: .;

Valid: LabelList (375806 items)
x: LMTextList
xxbos xxmaj why do so many women become so rude and arrogant when they get just a little bit of wealth and power ?,xxbos xxmaj when should i apply for xxup rv college of engineering and xxup bms college of engineering ? xxmaj should i wait for the xxup comedk result or am i supposed to apply before the result ?,xxbos xxmaj what is it really like to be a nurse practitioner ?,xxbos xxmaj who are entrepreneurs ?,xxbos xxmaj is education really making good people nowadays ?
y: LMLabelList
,,,,
Path: .;

Test: None
```

The tokenized prepared data is based on a lot of research from the FastAI developers. To make this post a little bit complete, I am sharing some of the tokens definition as well.

- *xxunk* is for an unknown word (one that isn't present in the current vocabulary)
- *xxpad* is the token used for padding, if we need to regroup several texts of different lengths in a batch
- *xxbos* represents the beginning of a text in your dataset
- *xxmaj* is used to indicate the next word begins with a capital in the original text
- *xxup* is used to indicate the next word is written in all caps in the original text

#### b) Finetune Base Language Model on Task Specific Data

This task is also pretty easy when we look at the code. The specific details of how we do the training is what holds the essence.

```py
# Learning with Discriminative fine tuning
learn.fit_one_cycle(1, 1e-2)
learn.unfreeze()
learn.fit_one_cycle(1, 1e-3)
# Save encoder Object
learn.save_encoder('ft_enc')
```

The paper introduced two general concepts for this learning stage:

- **Discriminative fine-tuning:**

The Main Idea is: As different layers capture different types of information, they should be fine-tuned to different extents.
Instead of using the same learning rate for all layers of the model, discriminative fine-tuning allows us to tune each layer with different learning
rates. In the paper, the authors suggest first to finetune only the last layer, and then unfreeze all the layers with a learning rate lowered by a factor of 2.6.


- **Slanted triangular learning rates:**

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/nlp_tl/Stlr.png"  height="200" width="400" ></center>
</div>

According to the authors: *"For adapting its parameters to task-specific features, we would like the model to quickly converge to a suitable region of the parameter space in the beginning of training and then refine its parameters"*
The Main Idea is to use a high learning rate at the starting stage for increased learning and low learning rates to finetune at later stages in an epoch.

After training our Language model on the Quora dataset, we should be able to see how our model performs on the Language Model task itself. FastAI library provides us with a simple function to do that.

```py
# check how the language model performs
learn.predict("What should", n_words=10)
```
```
'What should be the likelihood of a tourist visiting Mumbai for'
```
---

#### c) Finetune Base Language Model Layers + Task Specific Layers on Task Specific Data

This is the stage where task-specific learning takes place that is we add the classification layers and fine tune them.

The authors augment the pretrained language model with two additional
linear blocks. Each block uses batch normalization (Ioffe and Szegedy, 2015) and dropout, with ReLU activations for the intermediate layer and a
softmax activation that outputs a probability distribution over target classes at the last layer. The params of these task-specific layers are the only ones that are learned from scratch.

```py
#Creating Classification Data
data_clas = TextClasDataBunch.from_df(path ="", train_df=train, valid_df =valid,  test_df=test_df, vocab=data_lm.train_ds.vocab, bs=32,label_cols = 'target')

# Creating Classifier Object
learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
# Add weights of finetuned Language model
learn.load_encoder('ft_enc')
# Fitting Classifier Object
learn.fit_one_cycle(1, 1e-2)
# Fitting Classifier Object after freezing all but last 2 layers
learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(5e-3/2., 5e-3))
# Fitting Classifier Object - discriminative learning
learn.unfreeze()
learn.fit_one_cycle(1, slice(2e-3/100, 2e-3))
```

Here also the Authors have derived a few novel methods:

- **Concat Pooling:**

The authors use not only the concatenation of all the hidden state but also the Maxpool and Meanpool representation of all hidden states as input to the linear layers.

$$ H = [h_1, . . . , h_T ] $$

$$ h_c = [h_T , maxpool(H), meanpool(H)] $$

- **Gradual Unfreezing:**

Rather than fine-tuning all layers at once, which risks catastrophic forgetting(Forgetting everything we have learned so far from language models), the authors propose to gradually unfreeze the model starting from the last layer as this contains the least general knowledge. The Authors first unfreeze the last layer and fine-tune all unfrozen layers for one epoch. They then unfreeze the next lower frozen layer and repeat, until they finetune all layers until convergence at the last iteration. The function `slice(2e-3/100, 2e-3)` means that we train every layer with different learning rates ranging from max to min value.

One can get the predictions for the test data at once using:

```py
test_preds = np.array(learn.get_preds(DatasetType.Test, ordered=True)[0])[:,1]
```

I am a big fan of Kaggle Kernels. One could not have imagined having all that compute for free. You can find a running version of the above code in this [kaggle kernel](https://www.kaggle.com/mlwhiz/ulmfit). Do try to experiment with it after forking and running the code. Also please upvote the kernel if you find it helpful.

----

## Results:

Here are the final results of all the different approaches I have tried on the Kaggle Dataset. I ran a 5 fold Stratified CV.

### a. Conventional Methods:

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/results_conv.png"  style="height:40%;width:40%"></center>
</div>

### b. Deep Learning Methods:

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/results_deep_learning.png"  style="height:50%;width:50%"></center>
</div>

### c. Transfer Learning Methods(ULMFIT):

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/nlp_tl/results_ulm.png"  style="height:30%;width:30%"></center>
</div>

The results achieved were not very good compared to deep learning methods, but I still liked the idea of the transfer learning approach, and it was so easy to implement it using fastAI. Also running the code took a lot of time at 9 hours, compared to other methods which got over in 2 hours.

Even if this approach didn't work well for this dataset, it is a valid approach for other datasets, as the Authors of the paper have achieved pretty good results on different datasets — definitely a genuine method to try out.


**PS:** Note that I didn't work on tuning the above models, so these results are only cursory. You can try to squeeze more performance by performing hyperparams tuning [using hyperopt](/blog/2017/12/28/hyperopt_tuning_ml_model/) or just old fashioned Grid-search.

----------
## Conclusion:

Finally, this post concludes my NLP Learning series. It took a lot of time to write, but the effort was well worth it. I hope you found it helpful in your work. I will try to write some more on this topic when I get some time. Follow me up at [Medium](https://mlwhiz.medium.com/) or Subscribe to my blog to be informed about my next posts.

If you want to know more about NLP, I would like to recommend this awesome [Natural Language Processing Specialization](https://coursera.pxf.io/9WjZo0). You can start for free with the 7-day Free Trial. This course covers a wide range of tasks in Natural Language Processing from basic to advanced: sentiment analysis, summarization, dialogue state tracking, to name a few.

Let me know if you think I can add something more to the post; I will try to incorporate it.

Cheers!!!
