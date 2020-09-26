---
title: "Today I Learned This Part I: What are word2vec Embeddings?"
date:  2017-04-09
draft: false
url : blog/2017/04/09/word_vec_embeddings_examples_understanding/
slug: word_vec_embeddings_examples_understanding
Category: Python, NLP, Algorithms, Kaggle ,TILT
Keywords:
- data scientist
- understand word2vec embeddings
- word2vec layman
- word2vec intuition
- word2vec
- today i learned this
- today i learned this series
- quora data science challenge
- question similarity

Tags:
- Deep Learning
- Artificial Intelligence
- Natural Language Processing

Categories:
- Natural Language Processing
- Deep Learning

description:  Your Daily dose of data science.This is a series of post in which I write about the things I learn almost everyday. This post particularly provides a description,examples,code and practical use cases for word2vec embeddings in real world.

toc : false

thumbnail: /images/category_bgs/default_bg.jpg
image: /images/category_bgs/default_bg.jpg
type : post
---

Recently Quora put out a [Question similarity](https://www.kaggle.com/c/quora-question-pairs) competition on Kaggle. This is the first time I was attempting an NLP problem so a lot to learn. The one thing that blew my mind away was the word2vec embeddings.

Till now whenever I heard the term word2vec I visualized it as a way to create a bag of words vector for a sentence.

For those who don't know *bag of words*:
If we have a series of sentences(documents)


1. This is good       - [1,1,1,0,0]
2. This is bad        - [1,1,0,1,0]
3. This is awesome    - [1,1,0,0,1]


Bag of words would encode it using *0:This 1:is 2:good 3:bad 4:awesome*

But it is much more powerful than that.

What word2vec does is that it creates vectors for words.
What I mean by that is that we have a 300 dimensional vector for every word(common bigrams too) in a dictionary.

## How does that help?

We can use this for multiple scenarios but the most common are:

A. *Using word2vec embeddings we can find out similarity between words*.
Assume you have to answer if these two statements signify the same thing:

1. President greets press in Chicago
2. Obama speaks to media in Illinois.

If we do a sentence similarity metric or a bag of words approach to compare these two sentences we will get a pretty low score.

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/word2vecembed.png"  height="400" width="700" ></center>
</div>

But with a word encoding we can say that


1. President is similar to Obama
2. greets is similar to speaks
3. press is similar to media
4. Chicago is similar to Illinois


B. *Encode Sentences*: I read a [post](https://www.linkedin.com/pulse/duplicate-quora-question-abhishek-thakur) from Abhishek Thakur a prominent kaggler.(Must Read). What he did was he used these word embeddings to create a 300 dimensional vector for every sentence.

His Approach: Lets say the sentence is "What is this"
And lets say the embedding for every word is given in 4 dimension(normally 300 dimensional encoding is given)

1. what : [.25 ,.25 ,.25 ,.25]
2. is   : [  1 ,  0 ,  0 ,  0]
3. this : [ .5 ,  0 ,  0 , .5]


Then the vector for the sentence is normalized elementwise addition of the vectors. i.e.


	Elementwise addition : [.25+1+0.5, 0.25+0+0 , 0.25+0+0, .25+0+.5] = [1.75, .25, .25, .75]
	divided by
	math.sqrt(1.25^2 + .25^2 + .25^2 + .75^2) = 1.5
	gives:[1.16, .17, .17, 0.5]


Thus I can convert any sentence to a vector  of a fixed dimension(decided by the embedding). To find similarity between two sentences I can use a variety of distance/similarity metrics.

C. Also It enables us to do algebraic manipulations on words which was not possible before. For example: What is king - man + woman ?

Guess what it comes out to be : *Queen*

## Application/Coding:

Now lets get down to the coding part as we know a little bit of fundamentals.

First of all we download a custom word embedding from Google. There are many other embeddings too.

```bash
wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
```

The above file is pretty big. Might take some time. Then moving on to coding.

```py
from gensim.models import word2vec
model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)
```

### 1. Starting simple, lets find out similar words. Want to find similar words to python?

```py
model.most_similar('python')
```
<div style="font-size:80%;color:black;font-family: helvetica;line-height:18px;margin-top:8px;margin-left:20px">
[(u'pythons', 0.6688377261161804),<br>
 (u'Burmese_python', 0.6680364608764648),<br>
 (u'snake', 0.6606293320655823),<br>
 (u'crocodile', 0.6591362953186035),<br>
 (u'boa_constrictor', 0.6443519592285156),<br>
 (u'alligator', 0.6421656608581543),<br>
 (u'reptile', 0.6387745141983032),<br>
 (u'albino_python', 0.6158879995346069),<br>
 (u'croc', 0.6083582639694214),<br>
 (u'lizard', 0.601341724395752)]<br>
 </div>

### 2. Now we can use this model to find the solution to the equation:
What is king - man + woman?

```py
model.most_similar(positive = ['king','woman'],negative = ['man'])
```
<div style="font-size:80%;color:black;font-family: helvetica;line-height:18px;margin-top:8px;margin-left:20px">
[(u'queen', 0.7118192315101624),<br>
 (u'monarch', 0.6189674139022827),<br>
 (u'princess', 0.5902431011199951),<br>
 (u'crown_prince', 0.5499460697174072),<br>
 (u'prince', 0.5377321839332581),<br>
 (u'kings', 0.5236844420433044),<br>
 (u'Queen_Consort', 0.5235946178436279),<br>
 (u'queens', 0.5181134343147278),<br>
 (u'sultan', 0.5098593235015869),<br>
 (u'monarchy', 0.5087412595748901)]<br>
</div>

You can do plenty of freaky/cool things using this:

### 3. Lets say you wanted a girl and had a girl name like emma in mind but you got a boy. So what is the male version for emma?

```py
model.most_similar(positive = ['emma','he','male','mr'],negative = ['she','mrs','female'])
```

<div style="font-size:80%;color:black;font-family: helvetica;line-height:18px;margin-top:8px;margin-left:20px">
[(u'sanchez', 0.4920658469200134),<br>
 (u'kenny', 0.48300960659980774),<br>
 (u'alves', 0.4684845209121704),<br>
 (u'gareth', 0.4530612826347351),<br>
 (u'bellamy', 0.44884198904037476),<br>
 (u'gibbs', 0.445194810628891),<br>
 (u'dos_santos', 0.44508373737335205),<br>
 (u'gasol', 0.44387346506118774),<br>
 (u'silva', 0.4424275755882263),<br>
 (u'shaun', 0.44144102931022644)]<br><br>
</div>

### 4. Find which word doesn't belong to a [list](https://github.com/dhammack/Word2VecExample/blob/master/main.py)?

```py
model.doesnt_match("math shopping reading science".split(" "))
```
I think staple doesn't belong in this list!


## Other Cool Things

### 1. Recommendations:

<div style="margin-top: 4px; margin-bottom: 10px;">
<center><img src="/images/recommendationpaper.png"  height="400" width="700" ></center>
</div>

In this [paper](https://arxiv.org/abs/1603.04259), the authors have shown that itembased CF can be cast in the same framework of word embedding.

### 2. Some other [examples](http://byterot.blogspot.in/2015/06/five-crazy-abstractions-my-deep-learning-word2doc-model-just-did-NLP-gensim.html) that people have seen after using their own embeddings:

Library - Books = Hall<br>
Obama + Russia - USA = Putin<br>
Iraq - Violence = Jordan<br>
President - Power = Prime Minister (Not in India Though)<br>

### 3.Seeing the above I started playing with it a little.

**Is this model sexist?**

```py
model.most_similar(positive = ["donald_trump"],negative = ['brain'])
```

<div style="font-size:80%;color:black;font-family: helvetica;line-height:18px;margin-top:8px;margin-left:20px">
[(u'novak', 0.40405112504959106),<br>
 (u'ozzie', 0.39440611004829407),<br>
 (u'democrate', 0.39187556505203247),<br>
 (u'clinton', 0.390536367893219),<br>
 (u'hillary_clinton', 0.3862358033657074),<br>
 (u'bnp', 0.38295692205429077),<br>
 (u'klaar', 0.38228923082351685),<br>
 (u'geithner', 0.380607008934021),<br>
 (u'bafana_bafana', 0.3801495432853699),<br>
 (u'whitman', 0.3790769875049591)]<br>
</div>

Whatever it is doing it surely feels like magic. Next time I will try to write more on how it works once I understand it fully.
