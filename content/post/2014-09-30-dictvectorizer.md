---
title: "Dictvectorizer for One Hot Encoding of Categorical Data"
date:  2014-09-30
draft: false
url : blog/2014/09/30/dictvectorizer_one_hot_encoding/
slug: dictvectorizer_one_hot_encoding
Category: Python
Keywords: 
- machine learning
Tags: 
- machine learning
description: How does a DictVectorizer works. There is a lot of stuff around the net for this but I dint get to understand much around it.
toc : false
---

## THE PROBLEM:

Recently I was working on the Criteo Advertising Competition on Kaggle. The competition was a classification problem which basically involved predicting the click through rates based on several features provided in the train data. Seeing the size of the data (11 GB Train), I felt that going with Vowpal Wabbit might be a better option.

But after getting to an CV error of .47 on the Kaggle LB and being stuck there , I felt the need to go back to Scikit learn. While SciKit learn seemed to have a partial_fit method in SGDClassifier, I still could not find a partial_fit method in the OneHotEncoder or DictVectorizer class which made me look to the internet again. Now while I could find many advices on how to use OneHotEncoding and DictVectorizer on small data, I cannot find something relate to data too big to store in the memory. How do I OneHotEncode such a large data file? 

## DICTVECTORIZER

How does a DictVectorizer works. There is a lot of stuff around the net for this but I dint get to understand much around it. This blog from Zygmuntz of Fastml came to rescue then. Although still it didn’t resolve how to apply that to such large amount of data. 


``` py
from sklearn.feature_extraction import DictVectorizer as DV
# Create Vectorizer
vectorizer = DV( sparse = False )
# Read the whole Data
traindata = pd.read_csv(train_file, header=None, sep=',', names = colnames)
# Retain the categorical Columns
train_df   = traindata[cat_col]
# Convert Panda Data frame to Dict
train_dict = train_df.T.to_dict().values()
# Create Fit
vectorizer.fit(test_dict)
```


## THE DATA

The data was basically comprised of 40 Features with: 1. First two Columns as ID, Label 2. Next 13 columns Continuous columns labelled I1-I13 3. Next 26 Columns Categorical labelled C1-C26 Further the categorical columns were very sparse and some of the categorical variables could take more than a million different values. 

## THE WORKAROUNDS

The main problem that I faced was that I could not fit that much data in a DataFrame, even when I have a machine of 16GB, and that lead me to think that do I have a need for such a large data frame. And that lead me to the first part of the solution. I don’t need to load the whole data at once. I just needed to create another dictionary with all the possible combinations and then fit my dictvectorizer on it. 

I know that it is a lot to take in, so let’s take an example to understand it: Let’s say we have a data of infinite size, which has 3 categorical variables: C1 could take values 1-100 C2 could take values 1-3 C3 could take values 1-1000 Then we just have to find which category could take the maximum number of values (i.e. C3 in the above case) and make a dict which contains other categories replicated to contain as many values In other words, we need to make a dict like: {C1 : [1,2,3,……,97,98,99,100]*10  , C2 : [1,2,3]*333+[1]  , C3: [1….1000]} Notice the star sign at the last of the list. That means that for every key in the dict the number of values is now 1000(i.e. the maximum number of features). 

And so that is what I did. After we have the Vectorizer Fit, the next task was to transform the data. I took the data transformed it and sent it to my model line by line. P.S. Don’t store the transformed data as around a 100000 records takes ~ 10GB of Hard Disk Space due to the high number of features. 

Hope you find it Informative and happy learning.

<script src="//z-na.amazon-adsystem.com/widgets/onejs?MarketPlace=US&adInstanceId=c4ca54df-6d53-4362-92c0-13cb9977639e"></script>
