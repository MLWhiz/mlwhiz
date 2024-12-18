---
title: "NLP  Learning Series: Part 2 - Conventional Methods for Text Classification"
date:  2019-02-08
draft: false
url : blog/2019/02/08/deeplearning_nlp_conventional_methods/
slug: deeplearning_nlp_conventional_methods
Category: ai, deep learning,kaggle, NLP
Keywords:
- artificial intelligence
-  deep learning methods for Nlp
- kaggle
-  pytorch
-  keras to pytorch
-  keras vs pytorch
-  keras is better than pytorch
-  pytorch better than keras
-  pytorch for text classification methods
-  kaggle text classification methods
-  text classification
-  NLP
-  preprocessing
-  preprocessing for NLP
-  Natural language processing methods
- artificial intelligence
- deep learning
- old methods nlp
- conventional methods nlp
- deep learning nlp pytorch
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

description: Recently, I started up with an NLP competition on Kaggle called Quora Question insincerity challenge. It is an NLP Challenge on text classification and as the problem has become more clear after working through the competition as well as by going through the invaluable kernels put up by the kaggle experts, I thought of sharing the knowledge. In this post, I will try to take you through some basic conventional models like TFIDF, Count Vectorizer, Hashing etc. that have been used in text classification and try to access their performance to create a baseline.
thumbnail : /images/tfidf.png
image : /images/tfidf.png
type : post
toc : false
---

This is the second post of the NLP Text classification series. To give you a recap, recently I started up with an NLP text classification competition on Kaggle called Quora Question insincerity challenge. And I thought to share the knowledge via a series of blog posts on text classification. The [first post](/blog/2019/01/17/deeplearning_nlp_preprocess/) talked about the various **preprocessing techniques that work with Deep learning models** and **increasing embeddings coverage**. In this post, I will try to take you through some **basic conventional models** like TFIDF, Count Vectorizer, Hashing etc. that have been used in text classification and try to access their performance to create a baseline. We will delve deeper into **Deep learning models** in the third post which will focus on different architectures for solving the text classification problem. We will try to use various other models which we were not able to use in this competition like **ULMFit transfer learning** approaches in the fourth post in the series.

**As a side note**: If you want to know more about NLP, I would like to recommend this awesome [Natural Language Processing Specialization](https://imp.i384100.net/555ABL). You can start for free with the 7-day Free Trial. This course covers a wide range of tasks in Natural Language Processing from basic to advanced: sentiment analysis, summarization, dialogue state tracking, to name a few.

It might take me a little time to write the whole series. Till then you can take a look at my other posts too: [What Kagglers are using for Text Classification](/blog/2018/12/17/text_classification/), which talks about various deep learning models in use in NLP and [how to switch from Keras to Pytorch](/blog/2019/01/06/pytorch_keras_conversion/).

So again we start with the first step: Preprocessing.

----------
## Basic Preprocessing Techniques for text data(Continued)

So in the last post, we talked about various preprocessing methods for text for deep learning purpose. Most of the preprocessing for conventional methods remains the same. **We will still remove special characters, punctuations, and contractions**. But We also may want to do stemming/lemmatization when it comes to conventional methods. Let us talk about them.

>For grammatical reasons, documents are going to use different forms of a word, such as organize, organizes, and organizing. Additionally, there are families of derivationally related words with similar meanings, such as democracy, democratic, and democratization.

Since we are going to create features for words in the feature creation step, it makes sense to reduce words to a common denominator so that 'organize','organizes' and 'organizing' could be referred to by a single word 'organize'

----------
### a) Stemming

Stemming is the process of converting words to their base forms using crude Heuristic rules. For example, one rule could be to remove 's' from the end of any word, so that 'cats' becomes 'cat'. or another rule could be to replace 'ies' with 'i' so that 'ponies becomes 'poni'. One of the main point to note here is that when we stem the word we might get a nonsense word like 'poni'. But it will still work for our use case as we count the number of occurrences of a particular word and not focus on the meanings of these words in conventional methods. It doesn't work with deep learning for precisely the same reason.

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/text_stemming.png"  style="height:90%;width:90%"></center>
</div>

We can do this pretty simply by using this function in python.

```py
from nltk.stem import  SnowballStemmer
from nltk.tokenize.toktok import ToktokTokenizer
def stem_text(text):
    tokenizer = ToktokTokenizer()
    stemmer = SnowballStemmer('english')
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens)
```

----------
### b) Lemmatization

Lemmatization is very similar to stemming but it aims to remove endings only if the base form is present in a dictionary.

```py
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.toktok import ToktokTokenizer
def lemma_text(text):
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    tokens = [wordnet_lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)
```

Once we are done with processing a text, our text will necessarily go through these following steps.

```py
def clean_sentence(x):
    x = x.lower()
    x = clean_text(x)
    x = clean_numbers(x)
    x = replace_typical_misspell(x)
    x = remove_stopwords(x)
    x = replace_contractions(x)
    x = lemma_text(x)
    x = x.replace("'","")
    return x
```

----------
## Text Representation

In Conventional Machine learning methods, we ought to create features for a text. There are a lot of representations that are present to achieve this. Let us talk about them one by one.

### a) Bag of Words - Countvectorizer Features

Suppose we have a series of sentences(documents)

```py
X = [
     'This is good',
     'This is bad',
     'This is awesome'
     ]
```

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/countvectorizer.png"  style="height:90%;width:90%"></center>
</div>

Bag of words will create a dictionary of the most common words in all the sentences. For the example above the dictionary would look like:


```py
word_index
{'this':0,'is':1,'good':2,'bad':3,'awesome':4}
```
And then encode the sentences using the above dict.

```py
This is good - [1,1,1,0,0]
This is bad - [1,1,0,1,0]
This is awesome - [1,1,0,0,1]
```

We could do this pretty simply in Python by using the CountVectorizer class from Python. Don't worry much about the heavy name, it just does what I explained above. It has a lot of parameters most significant of which are:

- **ngram_range:** I specify in the code (1,3). This means that unigrams, bigrams, and trigrams will be taken into account while creating features.
- **min_df:** Minimum no of time an ngram should appear in a corpus to be used as a feature.

```py
cnt_vectorizer = CountVectorizer(dtype=np.float32,
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3),min_df=3)


# we fit count vectorizer to get ngrams from both train and test data.
cnt_vectorizer.fit(list(train_df.cleaned_text.values) + list(test_df.cleaned_text.values))

xtrain_cntv =  cnt_vectorizer.transform(train_df.cleaned_text.values)
xtest_cntv = cnt_vectorizer.transform(test_df.cleaned_text.values)
```

We could then use these features with any machine learning classification model like Logistic Regression, Naive Bayes, SVM or LightGBM as we would like.
For example:

```py
# Fitting a simple Logistic Regression on CV Feats
clf = LogisticRegression(C=1.0)
clf.fit(xtrain_cntv,y_train)
```

[Here](https://www.kaggle.com/mlwhiz/conventional-methods-for-quora-classification/) is a link to a kernel where I tried these features on the Quora Dataset. If you like it please don't forget to upvote.

----------
### b) TFIDF Features

TFIDF is a simple technique to find features from sentences. While in Count features we take count of all the words/ngrams present in a document, with TFIDF we take features only for the significant words. How do we do that? If you think of a document in a corpus, we will consider two things about any word in that document:

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/tfidf.png"  style="height:90%;width:90%"></center>
</div>

- **Term Frequency:** How important is the word in the document?

$$TF(word\ in\ a\ document) = \dfrac{No\ of\ occurances\ of\ that\ word\ in\ document}{No\ of\ words\ in\ document}$$

- **Inverse Document Frequency:** How important the term is in the whole corpus?

$$IDF(word\ in\ a\ corpus) = -log(ratio\ of\ documents\ that\ include\ the\ word)$$

TFIDF then is just multiplication of these two scores.

Intuitively, One can understand that a word is important if it occurs many times in a document. But that creates a problem. Words like "a", "the" occur many times in sentence. Their TF score will always be high. We solve that by using Inverse Document frequency, which is high if the word is rare, and low if the word is common across the corpus.

In essence, we want to find important words in a document which are also not very common.

We could do this pretty simply in Python by using the TFIDFVectorizer class from Python. It has a lot of parameters most significant of which are:

- **ngram_range:** I specify in the code (1,3). This means that unigrams, bigrams, and trigrams will be taken into account while creating features.
- **min_df:** Minimum no of time an ngram should appear in a corpus to be used as a feature.

```py
# Always start with these features. They work (almost) everytime!
tfv = TfidfVectorizer(dtype=np.float32, min_df=3,  max_features=None,
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

# Fitting TF-IDF to both training and test sets (semi-supervised learning)
tfv.fit(list(train_df.cleaned_text.values) + list(test_df.cleaned_text.values))
xtrain_tfv =  tfv.transform(train_df.cleaned_text.values)
xvalid_tfv = tfv.transform(test_df.cleaned_text.values)

```
Again, we could use these features with any machine learning classification model like Logistic Regression, Naive Bayes, SVM or LightGBM as we would like. [Here](https://www.kaggle.com/mlwhiz/conventional-methods-for-quora-classification/) is a link to a kernel where I tried these features on the Quora Dataset. If you like it please don't forget to upvote.


----------
### c) Hashing Features

Normally there will be a lot of ngrams in a document corpus. The number of features that our TFIDFVectorizer generated was in excess of 2,00,000 features. This might lead to a problem on very large datasets as we have to hold a very large vocabulary dictionary in memory. One way to counter this is to use the Hash Trick.

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/hashfeats.png"  style="height:90%;width:90%"></center>
</div>

One can think of hashing as a single function which maps any ngram to a number range for example between 0 to 1024. Now we don't have to store our ngrams in a dictionary. We can just use the function to get the index of any word, rather than getting the index from a dictionary.

Since there can be more than 1024 ngrams, different ngrams might map to the same number, and this is called collision. The larger the range we provide our Hashing function, the less is the chance of collisions.

We could do this pretty simply in Python by using the HashingVectorizer class from Python. It has a lot of parameters most significant of which are:

- **ngram_range:** I specify in the code (1,3). This means that unigrams, bigrams, and trigrams will be taken into account while creating features.
- **n_features:** No of features you want to consider. The range I gave above.

```py
# Always start with these features. They work (almost) everytime!
hv = HashingVectorizer(dtype=np.float32,
            strip_accents='unicode', analyzer='word',
            ngram_range=(1, 4),n_features=2**12,non_negative=True)
# Fitting Hash Vectorizer to both training and test sets (semi-supervised learning)
hv.fit(list(train_df.cleaned_text.values) + list(test_df.cleaned_text.values))
xtrain_hv =  hv.transform(train_df.cleaned_text.values)
xvalid_hv = hv.transform(test_df.cleaned_text.values)
y_train = train_df.target.values
```
[Here](https://www.kaggle.com/mlwhiz/conventional-methods-for-quora-classification/) is a link to a kernel where I tried these features on the Quora Dataset. If you like it please don't forget to upvote.


----------
### d) Word2vec Features

We already talked a little about word2vec in the previous post. We can use the word to vec features to create sentence level feats also. We want to create a `d` dimensional vector for sentence. For doing this, we will simply average the word embedding of all the words in a sentence.

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/word2vec_feats.png"  style="height:90%;width:90%"></center>
</div>

We can do this in Python using the following functions.
```py
# load the GloVe vectors in a dictionary:
def load_glove_index():
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')[:300]
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))
    return embeddings_index

embeddings_index = load_glove_index()

print('Found %s word vectors.' % len(embeddings_index))

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())

# create glove features
xtrain_glove = np.array([sent2vec(x) for x in tqdm(train_df.cleaned_text.values)])
xtest_glove = np.array([sent2vec(x) for x in tqdm(test_df.cleaned_text.values)])
```

[Here](https://www.kaggle.com/mlwhiz/conventional-methods-for-quora-classification/) is a link to a kernel where I tried these features on the Quora Dataset. If you like it please don't forget to upvote.

----------
## Results
Here are the results of different approaches on the Kaggle Dataset. I ran a 5 fold Stratified CV.

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/results_conv.png"  style="height:40%;width:40%"></center>
</div>




[Here](https://www.kaggle.com/mlwhiz/conventional-methods-for-quora-classification/) is the code. If you like it please don't forget to upvote.
Also note that I didn't work on tuning the models, so these results are only cursory. You can try to squeeze more performance by performing hyperparams tuning [using hyperopt](/blog/2017/12/28/hyperopt_tuning_ml_model/) or just old fashioned Grid-search and the performance of models may change after that substantially.

----------
## Conclusion

While Deep Learning works a lot better for NLP classification task, it still makes sense to have an understanding of how these problems were solved in the past, so that we can appreciate the nature of the problem. I have tried to provide a perspective on the conventional methods and one should experiment with them too to create baselines before moving to Deep Learning methods. If you want to know more about NLP, I would like to recommend this awesome [Natural Language Processing Specialization](https://imp.i384100.net/555ABL). You can start for free with the 7-day Free Trial. This course covers a wide range of tasks in Natural Language Processing from basic to advanced: sentiment analysis, summarization, dialogue state tracking, to name a few.


----------
## Endnotes and References

This post is a result of an effort of a lot of excellent Kagglers and I will try to reference them in this section. If I leave out someone, do understand that it was not my intention to do so.

- [Approaching (Almost) Any NLP Problem on Kaggle](https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle)
- [How to: Preprocessing when using embeddings](https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings)

----------
