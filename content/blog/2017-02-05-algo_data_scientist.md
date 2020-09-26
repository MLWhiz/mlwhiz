---
title: "Machine Learning Algorithms for Data Scientists"
date:  2017-02-05
draft: false
url : blog/2017/02/05/ml_algorithms_for_data_scientist/
slug: ml_algorithms_for_data_scientist
aliases:
- blog/2017/02/05/Machine_learning_algorithms_for_data_scientist/

Category: Python, Algorithms, Statistics
Keywords:
- Python
- data scientist
- algorithms
Tags:
- Algorithms
- Statistics
- Machine Learning

description:  Machine Learning algorithms for Data Scientists
toc : false
Categories:
- Data Science
type : post
thumbnail: /images/category_bgs/default_bg.jpg
image: /images/category_bgs/default_bg.jpg
---



As a data scientist I believe that a lot of work has to be done before Classification/Regression/Clustering methods are applied to the data you get. The data which may be messy, unwieldy and big. So here are the list of algorithms that helps a data scientist to make better models using the data they have:

## 1. Sampling Algorithms. In case you want to work with a sample of data.

- **Simple Random Sampling :** *Say you want to select a subset of a population in which each member of the subset has an equal probability of being chosen.*
- **Stratified Sampling:** Assume that we need to estimate average number of votes for each candidate in an election. Assume that country has 3 towns : Town A has 1 million factory workers, Town B has 2 million workers and Town C has 3 million retirees. We can choose to get a random sample of size 60 over entire population but there is some chance that the random sample turns out to be not well balanced across these towns and hence is biased causing a significant error in estimation. Instead if we choose to take a random sample of 10, 20 and 30 from Town A, B and C respectively then we can produce a smaller error in estimation for the same total size of sample.
- **Reservoir Sampling** :*Say you have a stream of items of large and unknown length that we can only iterate over once. Create an algorithm that randomly chooses an item from this stream such that each item is equally likely to be selected.*


## 2. **Map-Reduce. If you want to work with the whole data**.

Can be used for feature creation. For Example: I had a use case where I had a graph of 60 Million customers and 130 Million accounts. Each account was connected to other account if they had the Same SSN or Same Name+DOB+Address. I had to find customer ID’s for each of the accounts. On a single node parsing such a graph took more than 2 days. On a Hadoop cluster of 80 nodes running a *Connected Component Algorithm* took less than 24 minutes. On Spark it is even faster.



## 3. **Graph Algorithms.**

Recently I was working on an optimization problem which was focussed on finding shortest distance and routes between two points in a store layout. Routes which don’t pass through different aisles, so we cannot use euclidean distances. We solved this problem by considering turning points in the store layout and the *djikstra’s Algorithm.*

<script src="//z-na.amazon-adsystem.com/widgets/onejs?MarketPlace=US&adInstanceId=c4ca54df-6d53-4362-92c0-13cb9977639e"></script>

## 4. **Feature Selection.**

- **Univariate Selection.** Statistical tests can be used to select those features that have the strongest relationship with the output variable.
- **VarianceThreshold.** Feature selector that removes all low-variance features.
- **Recursive Feature Elimination.** The goal of recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and weights are assigned to each one of them. Then, features whose absolute weights are the smallest are pruned from the current set features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached.
- **Feature Importance:** Methods that use ensembles of decision trees (like Random Forest or Extra Trees) can also compute the relative importance of each attribute. These importance values can be used to inform a feature selection process.



## 5. **Algorithms to work efficiently.**

Apart from these above algorithms sometimes you may need to write your own algorithms. Now I think of big algorithms as a combination of small but powerful algorithms. You just need to have idea of these algorithms to make a more better/efficient product. So some of these powerful algorithms which can help you are:

- **Recursive Algorithms:**Binary search algorithm.
- **Divide and Conquer Algorithms:** Merge-Sort.
- **Dynamic Programming:**Solving a complex problem by breaking it down into a collection of simpler subproblems, solving each of those subproblems just once, and storing their solutions.



## 6. **Classification/Regression Algorithms.** The usual suspects. Minimum you must know:

1. **Linear Regression -** Ridge Regression, Lasso Regression, ElasticNet
2. **Logistic Regression**
3. From there you can build upon:
    1. **Decision Trees -** ID3, CART, C4.5, C5.0
    2. **KNN**
    3. **SVM**
    4. **ANN** - Back Propogation, CNN
4. And then on to Ensemble based algorithms:
    1. **Boosting**: Gradient Boosted Trees
    2. **Bagging**: Random Forests
    3. **Blending**: Prediction outputs of different learning algorithms are fed into another learning algorithm.



## 7 . **Clustering Methods.**For unsupervised learning.

1. **k-Means**
2. **k-Medians**
3. **Expectation Maximisation (EM)**
4. **Hierarchical Clustering**



## 8. **Other algorithms you can learn about:**

1. **Apriori algorithm**- Association Rule Mining
2. **Eclat algorithm -** Association Rule Mining
3. **Item/User Based Similarity -** Recommender Systems
4. **Reinforcement learning -** Build your own robot.
5. **Graphical Models**
6. **Bayesian Algorithms**
7. **NLP -** For language based models. Chatbots.

Hope this has been helpful.....
