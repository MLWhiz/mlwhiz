---
title:   An Understandable Guide to ROC Curves And AUC and Why and When to use them?
date:  2021-02-04
draft: false
url : blog/2021/02/03/roc-auc-curves-explained/
slug: roc-auc-curves-explained

Keywords:
- ROC AUC

Tags:
- Statistics
- Machine Learning

Categories:
- Data Science


description: ROC curves, or receiver operating characteristic curves, are one of the most common evaluation metrics for checking a classification model’s performance. Unfortunately, many data scientists often just end up seeing the ROC curves and then quoting an AUC (short for the area under the ROC curve) value without really understanding what the AUC value means and how they can use them more effectively.Other times, they don’t understand the various problems that ROC curves solve and the multiple properties of AUC like threshold invariance and scale invariance, which necessarily means that the AUC metric doesn’t depend on the chosen threshold or the scale of probabilities. These properties make AUC pretty valuable for evaluating binary classifiers as it provides us with a way to compare them without caring about the classification threshold. That’s why it’s important for data scientists to have a fuller understanding of both ROC curves and AUC.

thumbnail : /images/roc-auc-curves-explained/main.png
image : /images/roc-auc-curves-explained/main.png
toc : false
type : "post"
---


ROC curves, or receiver operating characteristic curves, are one of the most common evaluation metrics for checking a classification model’s performance. Unfortunately, many data scientists often just end up seeing the ROC curves and then quoting an AUC (short for the area under the ROC curve) value without really understanding what the AUC value means and how they can use them more effectively.

Other times, they don’t understand the various problems that ROC curves solve and the multiple properties of AUC like threshold invariance and scale invariance, which necessarily means that the AUC metric doesn’t depend on the chosen threshold or the scale of probabilities. These properties make AUC pretty valuable for evaluating binary classifiers as it provides us with a way to compare them without caring about the classification threshold. That’s why it’s important for data scientists to have a fuller understanding of both ROC curves and AUC.

---
## So, what need do ROC Curves/AUC fulfil?

So, the first question that comes to mind before we even start to learn about ROC curves and AUC is why not use a fairly simple metric like accuracy for a binary classification task? After all, accuracy is pretty easy to understand as it just calculates the percentage of correct predictions by a model.

The answer is that accuracy doesn’t capture the whole essence of a probabilistic classifier, i.e., it is neither a threshold-invariant metric nor a scale-invariant metric. What do I mean by that? Better to explain using some examples.

**1. Why is Accuracy not threshold-invariant?**

Assuming a threshold of 0.5 for a logistic regression classifier, what do you think the accuracy of this classifier is?

![Source: Image by Author](/images/roc-auc-curves-explained/0.png)*Source: Image by Author*

If you said 50 per cent, congratulations. We would misclassify the two zeroes as ones. This result isn’t that great. But is our classifier really that bad? Based on accuracy as an evaluation metric, it seems that it is. But what if we change the threshold in the same example to 0.75? Now, our classifier becomes 100 per cent accurate. **This should lead us to ask how we can come up with an evaluation metric that doesn’t depend on the threshold**. That is, we want a threshold-invariant metric.

**2. Why is Accuracy not scale-invariant?**

Now, let’s do the same exercise again, but this time our classifier predicts different probabilities but in the **same rank order**. This means that the probability values change, but the order remains the same. So in Classifier B, the rank of predictions remains the same while Classifier C predicts on a whole different scale. So, which of the following is the best?

![Source: Image by Author](/images/roc-auc-curves-explained/1.png)*Source: Image by Author*

In all these cases, we can see that each classifier is largely the same. That is, if we have a threshold of 0.75 for Classifier A, 0.7 for Classifier B and 68.5 for Classifier C, we have a 100 per cent accuracy on all of them.

The property of having the same value for an evaluation metric when the rank order remains the same is called the scale-invariant property. This property can really help us in cases where a classifier predicts a score rather than a probability, thereby allowing us to compare two different classifiers that predict values on a different scale.

So, finally, we want an evaluation metric that satisfies the following two conditions:

* It is **threshold invariant** **i.e.** the value of the metric doesn’t depend on a chosen threshold.

* It is **scale-invariant i.e. **It measures how well predictions are ranked, rather than their absolute values.

The excellent news is that AUC fulfils both the above conditions. Before we can even look at how AUC is calculated, however, let’s understand ROC curves in more detail.

---
## ROC Curves

A quick historical fun fact about ROC curves is that they were first used during World War II for the analysis of radar signals. After the attacks on Pearl Harbor, the United States military wanted to detect Japanese aircraft using their radar signals. ROC curves were particularly good for this task as they let the operators choose thresholds for discriminating positive versus negative examples.

But how do we make these curves ourselves? To understand this, we need to understand **true positive rate** (TPR) and **false positive rate** (FPR) first. So, let’s say we have the following sample confusion matrix for a model with a particular probability threshold:

![Source: Author Image](/images/roc-auc-curves-explained/2.png)*Source: Author Image*

To explain TPR and FPR, I usually give the example of a justice system. Naturally, any justice system only wants to punish people guilty of crimes and doesn’t want to charge an innocent person. Now let’s say that the model above is a justice system that evaluates each citizen and predicts either a zero (innocent) or a one (guilty).

In this case, the TPR is the proportion of guilty criminals our model was able to capture. Thus, the numerator is guilty criminals captured, and the denominator is total criminals. This ratio is also known as recall or sensitivity.

**TPR(True Positive Rate)/Sensitivity/Recall**= TP/(TP+FN)

The FPR is the proportion of innocents we incorrectly predicted as criminal (false positives) divided by the total number of actual innocent citizens. Thus, the numerator is innocents captured, and the denominator is total innocents.

**FPR(False Positive Rate)**= FP/(TN+FP)

Usually, we would want high TPR (because we want to capture all the criminals) and low FPR (because we don’t want to capture innocent people).

So, how do we plot ROC Curves using TPR and FPR? We plot false positive rate (FPR) on the X-axis vs true positive rate (TPR) on the Y-axis using different threshold values. The resulting curve when we join these points is called the ROC Curve.

Let’s go through a simple code example here to understand how to do this in Python. Below, we just create a small sample classification data set and fit a logistic regression model on the data. We also get the probability values from the classifier.

    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    import plotly.express as px
    import pandas as pd

    # Random Classification dataset
    X, y = make_classification(n_samples=1000, n_classes=2, random_state=1)

    model = LogisticRegression()
    model.fit(X, y)

    # predict probabilities
    preds = model.predict_proba(X)[:,1]

Now we want to evaluate how good our model is using ROC curves. To do this, we need to find FPR and TPR for various threshold values. We can do this pretty easily by using the function roc_curve from sklearn.metrics, which provides us with FPR and TPR for various threshold values as shown below:

    fpr, tpr, thresh = roc_curve(y, preds)

    roc_df = pd.DataFrame(zip(fpr, tpr, thresh),columns = ["FPR","TPR","Threshold"])

![We start by getting FPR and TPR for various threshold values. Source: Author Image](/images/roc-auc-curves-explained/3.png)*We start by getting FPR and TPR for various threshold values. Source: Author Image*

Now all that remains is plotting the curve using the above data. We can do this by using any graphing library, but I prefer [plotly.express](https://mlwhiz.com/blog/2019/05/05/plotly_express/) as it is pretty easy to use and even allows you to use plotly constructs on top of plotly express figures. As you can see in the below curve, we plotted FPR vs TPR for various threshold values.

![Source: Image by Author](/images/roc-auc-curves-explained/4.png)*Source: Image by Author*

### **How to use the ROC Curve?**

We can generally use ROC curves to decide on a threshold value. The choice of threshold value will also depend on how the classifier is intended to be used. So, if the above curve was for a cancer prediction application, you want to capture the maximum number of positives (i.e., have a high TPR) and you might choose a low value of threshold like 0.16 even when the FPR is pretty high here.

This is because you really don’t want to predict “no cancer” for a person who actually has cancer. In this example, the cost of a false negative is pretty high. You are OK even if a person who doesn’t have cancer tests positive because the cost of false positive is lower than that of a false negative. This is actually what a lot of clinicians and hospitals do for such vital tests and also why a lot of clinicians do the same test for a second time if a person tests positive. (Can you think why doing so helps? Hint: Bayes Rule).

Otherwise, in a case like the criminal classifier from the previous example, we don’t want a high FPR as one of the tenets of the justice system is that we don’t want to capture any innocent people. So, in this case, we might choose our threshold as 0.82, which gives us a good recall or TPR of 0.6. That is, we can capture 60 per cent of criminals.

---
## **Now, what is AUC?**

The AUC is the area under the ROC Curve. This area is always represented as a value between 0 to 1 (just as both TPR and FPR can range from 0 to 1), and we essentially want to maximize this area so that we can have the highest TPR and lowest FPR for some threshold.

Scikit also provides a utility function that lets us get AUC if we have predictions and actual y values using roc_auc_score(y, preds).

![[Source](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#/media/File:Roc-draft-xkcd-style.svg): Wikipedia](/images/roc-auc-curves-explained/5.png)*[Source](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#/media/File:Roc-draft-xkcd-style.svg): Wikipedia*

It can also be mathematically proven that AUC is equal to the probability that a classifier will rank a randomly chosen positive instance higher than a randomly chosen negative one. Thus, an AUC of 0.5 means that the probability of a positive instance ranking higher than a negative instance is 0.5 and hence random. A perfect classifier would always rank a positive instance higher than a negative one and have an AUC of 1.

### So, is AUC threshold-invariant and Scale-Invariant?

Yes, AUC is threshold-invariant as we don’t have to set a threshold to calculate AUC.

For checking scale invariance, I will essentially do an experiment in which I multiply our predictions by a random factor (scaling) and also exponentiate the predictions to check whether the AUC changes if the predictions change even though their rank-order doesn’t change. As AUC is scale-invariant, I would expect the same ROC curve and same AUC metric. Below, you can see the scaling on the left and exponential rank order on the right.

![](/images/roc-auc-curves-explained/6.png)

![Scaling(Left) and Exponentiation Rank Order(Right)](/images/roc-auc-curves-explained/7.png)*Scaling(Left) and Exponentiation Rank Order(Right)*

And that is in fact what I got. Only the threshold changes as the scale changes. The shape of the curve, as well as the AUC, remains precisely the same.

---
## Conclusion

An important step while creating any [Machine learning pipeline](https://towardsdatascience.com/6-important-steps-to-build-a-machine-learning-system-d75e3b83686) is evaluating different models against each other. Picking the wrong evaluation metric or not understanding what your metric really means could wreak havoc to your whole system. I hope that, with this post, I was able to clear some confusion that you might have had with ROC curves and AUC.

If you want to learn more about how to structure a machine learning project and the best practices for doing so, I would recommend this excellent [third course](https://click.linksynergy.com/link?id=lVarvwc5BD0&offerid=467035.11421702016&type=2&murl=https%3A%2F%2Fwww.coursera.org%2Flearn%2Fmachine-learning-projects) named “Structuring Machine Learning Projects” in the Coursera [Deep Learning Specialization](https://click.linksynergy.com/deeplink?id=lVarvwc5BD0&mid=40328&murl=https%3A%2F%2Fwww.coursera.org%2Fspecializations%2Fdeep-learning). Do check it out. It addresses the pitfalls and a lot of basic ideas to improve your models.

I am going to be writing more of such posts in the future too. Let me know what you think about the series. Follow me up at [**Medium](http://mlwhiz.medium.com)** or Subscribe to my [**blog](https://mlwhiz.ck.page/a9b8bda70c)** to be informed about them. As always, I welcome feedback and constructive criticism and can be reached on Twitter [@mlwhiz](https://twitter.com/MLWhiz)

Also, a small disclaimer — There might be some affiliate links in this post to relevant resources, as sharing knowledge is never a bad idea.
