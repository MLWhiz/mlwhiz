---
title: "The story of every distribution - Discrete Distributions"
date:  2017-09-14
draft: false
url : blog/2017/09/14/discrete_distributions/
slug: discrete_distributions
Category: statistics,probability
Keywords:
- distributions
-  pdf
-  cdf
-  expected value
-  variance
-  binomial
- poisson
-  geometric
-  cdf proof
-  pdf proof
-  story
-  expected value proof
- deeplearning
- pretrained models
- use pretrained neural nets for classifying images
- deep learning
- kaggle competition dogs vs cats
- neural neworks pretrained

Categories:
- Data Science

Tags:
- Statistics
- Data Science
description:  Distributions play an important role in the life of every Statistician. Properties of famous discrete distributions
toc : false
thumbnail: /images/output_14_0.png
image: /images/output_14_0.png
type : post  
---

Distributions play an important role in the life of every Statistician. I coming from a non-statistic background am not so well versed in these and keep forgetting about the properties of these famous distributions. That is why I chose to write my own understanding in an intuitive way to keep a track.
One of the most helpful way to learn more about these is the [STAT110](https://projects.iq.harvard.edu/stat110/home) course by Joe Blitzstein and his [book](http://amzn.to/2xAsYzE). You can check out this [Coursera](https://imp.i384100.net/N99zYK) course too. Hope it could be useful to someone else too. So here goes:

## 1. Bernoulli Distribution:

Perhaps the most simple discrete distribution of all.

**Story:** A Coin is tossed with probability p of heads.

**PMF of Bernoulli Distribution is given by:**

<div>$$P(X=k) = \begin{cases}1-p & k = 0\\p & k = 1\end{cases}$$</div>

**CDF of Bernoulli Distribution is given by:**

<div>$$P(X \leq k) = \begin{cases}0 & k \lt 0\\1-p & 0 \leq k \lt 1 \\1 & k \geq 1\end{cases}$$</div>

<strong>Expected Value:</strong>

$$E[X] = \sum kP(X=k)$$
$$E[X] = 0*P(X=0)+1*P(X=1) = p$$

<strong>Variance:</strong>

$$Var[X] = E[X^2] - E[X]^2$$
Now we find,
$$E[X]^2 = p^2$$
and
$$E[X^2] = \sum k^2P(X=k)$$
$$E[X^2] =  0^2P(X=0) + 1^2P(X=1) = p $$
Thus,
$$Var[X] = p(1-p)$$

## 2. Binomial Distribution:

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/maxresdefault.jpg"  height="400" width="500" ></center>
</div>

One of the most basic distribution in the Statistician toolkit. The parameters of this distribution is n(number of trials) and p(probability of success).

**Story:**
Probability of getting exactly k successes in n trials

**PMF of binomial Distribution is given by:**

$$P(X=k) = \left(\begin{array}{c}n\\ k\end{array}\right) p^{k}(1-p)^{n-k}$$

**CDF of binomial Distribution is given by:**

$$ P(X\leq k) = \sum_{i=0}^k  \left(\begin{array}{c}n\\ i\end{array}\right)  p^i(1-p)^{n-i} $$

<strong>Expected Value:</strong>

$$E[X] = \sum kP(X=k)$$
$$E[X] = \sum_{k=0}^n k \left(\begin{array}{c}n\\ k\end{array}\right) * p^{k}(1-p)^{n-k} = np $$

A better way to solve this:

<div>$$ X = I_{1} + I_{2} + ....+ I_{n-1}+ I_{n} $$</div>

X is the sum on n Indicator Bernoulli random variables.

Thus,

<div>
$$E[X] = E[I_{1} + I_{2} + ....+ I_{n-1}+ I_{n}]$$</div>

<div>$$E[X] = E[I_{1}] + E[I_{2}] + ....+ E[I_{n-1}]+ E[I_{n}]$$</div>

<div>$$E[X] = \underbrace{p + p + ....+ p + p}_{n} = np$$</div>

<strong>Variance:</strong>

<div>$$ X = I_{1} + I_{2} + ....+ I_{n-1}+ I_{n} $$</div>
X is the sum on n Indicator Bernoulli random variables.
<div>$$Var[X] = Var[I_{1} + I_{2} + ....+ I_{n-1}+ I_{n}]$$</div>
<div>$$Var[X] = Var[I_{1}] + Var[I_{2}] + ....+ Var[I_{n-1}]+ Var[I_{n}]$$</div>
<div>$$Var[X] = \underbrace{p(1-p) + p(1-p) + ....+ p(1-p) + p(1-p)}_{n} = np(1-p)$$</div>


## 3. Geometric Distribution:

The parameters of this distribution is p(probability of success).

**Story:**
The number of failures before the first success(Heads) when a coin with probability p is tossed

**PMF of Geometric Distribution is given by:**

$$P(X=k) = (1-p)^kp$$

**CDF of Geometric Distribution is given by:**

$$ P(X\leq k) = \sum_{i=0}^k (1-p)^{i}p$$
$$ P(X\leq k) = p(1+q+q^2...+q^k)= p(1-q^k)/(1-q) = 1-(1-p)^k $$

<strong>Expected Value:</strong>

$$E[X] = \sum kP(X=k)$$
$$E[X] = \sum_{k=0}^{inf} k (1-p)^kp$$
$$E[X] = qp +2q^2p +3q^3p +4q^4p .... $$
$$E[X] = qp(1+2q+3q^2+4q^3+....)$$
$$E[X] = qp/(1-q)^2 = q/p $$


<strong>Variance:</strong>

$$Var[X] = E[X^2] - E[X]^2$$
Now we find,
$$E[X]^2 = q^2/p^2$$
and
$$E[X^2] = \sum_{k=0}^{inf} k^2q^kp= qp + 4q^2p + 9q^3p +16q^4p ... = qp(1+4q+9q^2+16q^3....)$$
$$E[X^2] = qp^{-2}(1+q)$$

Thus,
$$Var[X] =q/p^2$$

Check Math appendix at bottom of this post for Geometric Series Proofs.

<strong>Example:</strong>

Q. A doctor is seeking an anti-depressant for a newly diagnosed patient. Suppose that, of the available anti-depressant drugs, the probability that any particular drug will be effective for a particular patient is p=0.6. What is the probability that the first drug found to be effective for this patient is the first drug tried, the second drug tried, and so on? What is the expected number of drugs that will be tried to find one that is effective?

A. Expected number of drugs that will be tried to find one that is effective = q/p = .4/.6 =.67

## 4. Negative Binomial Distribution:

The parameters of this distribution is p(probability of success) and r(number of success).

**Story:**
The **number of failures** of independent Bernoulli(p) trials before the rth success.

**PMF of Negative Binomial Distribution is given by:**

r successes , k failures , last attempt needs to be a success:
$$P(X=k) = \left(\begin{array}{c}k+r-1\\ k\end{array}\right) p^r(1-p)^k$$

<strong>Expected Value:</strong>

The negative binomial RV could be stated as the sum of r Geometric RVs
$$X = X^1+X^2.... X^{r-1} +X^r$$
Thus,
$$E[X] = E[X^1]+E[X^2].... E[X^{r-1}] +E[X^r]$$

$$E[X] = rq/p$$

<strong>Variance:</strong>

The negative binomial RV could be stated as the sum of r independent Geometric RVs
$$X = X^1+X^2.... X^{r-1} +X^r$$
Thus,
$$Var[X] = Var[X^1]+Var[X^2].... Var[X^{r-1}] +Var[X^r]$$

$$Var[X] = rq/p^2$$

<strong>Example:</strong>

Q. Pat is required to sell candy bars to raise money for the 6th grade field trip. There are thirty houses in the neighborhood, and Pat is not supposed to return home until five candy bars have been sold. So the child goes door to door, selling candy bars. At each house, there is a 0.4 probability of selling one candy bar and a 0.6 probability of selling nothing.
What's the probability of selling the last candy bar at the nth house?

A. r = 5 ; k = n - r

Probability of selling the last candy bar at the nth house =
$$P(X=k) = \left(\begin{array}{c}k+r-1\\ k\end{array}\right) p^r(1-p)^k$$
$$P(X=k) = \left(\begin{array}{c}n-1\\ n-5\end{array}\right) .4^5(.6)^{n-5}$$

## 5. Poisson Distribution:

The parameters of this distribution is $\lambda$ the rate parameter.

**Motivation:**
There is as such no story to this distribution but only motivation for using this distribution. The Poisson distribution is often used for applications where we count the successes of a large number of trials where the per-trial success rate is small. For example, the Poisson distribution is a good starting point for counting the number of people who email you over the course of an hour.The number of chocolate chips in a chocolate chip cookie is another good candidate for a Poisson distribution, or the number of earthquakes in a year in some particular region

**PMF of Poisson Distribution is given by:**
$$ P(X=k) = \frac{e^{-\lambda}\lambda^k} {k!}$$

<strong>Expected Value:</strong>

<div>$$E[X] = \sum kP(X=k)$$</div>

<div>$$ E[X] = \sum_{k=0}^{inf} k \frac{e^{-\lambda}\lambda^k} {k!}$$</div>
<div>$$ E[X] = \lambda e^{-\lambda}\sum_{k=0}^{inf}  \frac{\lambda^{k-1}} {(k-1)!}$$</div>
<div>$$ E[X] = \lambda e^{-\lambda} e^{\lambda} = \lambda $$</div>

<strong>Variance:</strong>

<div>$$Var[X] = E[X^2] - E[X]^2$$</div>

Now we find,
<div>$$E[X^2] = \lambda + \lambda^2$$</div>
Thus,
$$Var[X] = \lambda$$

<strong>Example:</strong>

Q. If electricity power failures occur according to a Poisson distribution with an average of 3 failures every twenty weeks, calculate the probability that there will not be more than one failure during a particular week?

A. Probability = P(X=0)+P(X=1) =

<div>$$e^{-3/20} + e^{-3/20}3/20 = 23/20*e^{-3/20} $$</div>

Probability of selling the last candy bar at the nth house =
$$P(X=k) = \left(\begin{array}{c}k+r-1\\ k\end{array}\right) p^r(1-p)^k$$
$$P(X=k) = \left(\begin{array}{c}n-1\\ n-5\end{array}\right) .4^5(.6)^{n-5}$$

## Math Appendix:

Some Math (For Geometric Distribution) :

$$a+ar+ar^2+ar^3+⋯=a/(1−r)=a(1−r)^{−1}$$
Taking the derivatives of both sides, the first derivative with respect to r must be:
$$a+2ar+3ar^2+4ar^3⋯=a(1−r)^{−2}$$
Multiplying above with r:
$$ar+2ar^2+3ar^3+4ar^4⋯=ar(1−r)^{−2}$$
Taking the derivatives of both sides, the first derivative with respect to r must be:
$$a+4ar+9ar^2+16ar^3⋯=a(1−r)^{-3}(1+r)$$

## Bonus - Python Graphs and Functions:

```py
# Useful Function to create graph
def chart_creator(x,y,title):
    import matplotlib.pyplot as plt  #sets up plotting under plt
    import seaborn as sns           #sets up styles and gives us more plotting options
    import pandas as pd             #lets us handle data as dataframes
    %matplotlib inline
    # Create a list of 100 Normal RVs
    data = pd.DataFrame(zip(x,y))
    data.columns = ['x','y']
    # We dont Probably need the Gridlines. Do we? If yes comment this line
    sns.set(style="ticks")

    # Here we create a matplotlib axes object. The extra parameters we use
    # "ci" to remove confidence interval
    # "marker" to have a x as marker.
    # "scatter_kws" to provide style info for the points.[s for size]
    # "line_kws" to provide style info for the line.[lw for line width]

    g = sns.regplot(x='x', y='y', data=data, ci = False,
        scatter_kws={"color":"darkred","alpha":0.3,"s":90},
        line_kws={"color":"g","alpha":0.5,"lw":0},marker="x")

    # remove the top and right line in graph
    sns.despine()

    # Set the size of the graph from here
    g.figure.set_size_inches(12,8)
    # Set the Title of the graph from here
    g.axes.set_title(title, fontsize=34,color="r",alpha=0.5)
    # Set the xlabel of the graph from here
    g.set_xlabel("k",size = 67,color="r",alpha=0.5)
    # Set the ylabel of the graph from here
    g.set_ylabel("pmf",size = 67,color="r",alpha=0.5)
    # Set the ticklabel size and color of the graph from here
    g.tick_params(labelsize=14,labelcolor="black")
```

And here I will generate the PMFs of the discrete distributions we just discussed above using Pythons built in functions. For more details on the upper function, please see my previous post - [Create basic graph visualizations with SeaBorn](http://mlwhiz.com/blog/2015/09/13/seaborn_visualizations/). Also take a look at the [documentation](https://docs.scipy.org/doc/scipy/reference/stats.html) guide for the below functions

```py
# Binomial :
from scipy.stats import binom
n=30
p=0.5
k = range(0,n)
pmf = binom.pmf(k, n, p)
chart_creator(k,pmf,"Binomial PMF")
```

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/output_12_0.png"  height="400" width="700" ></center>
</div>

```py
# Geometric :
from scipy.stats import geom
n=30
p=0.5
k = range(0,n)
# -1 here is the location parameter for generating the PMF we want.
pmf = geom.pmf(k, p,-1)
chart_creator(k,pmf,"Geometric PMF")
```

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/output_13_0.png"  height="400" width="700" ></center>
</div>


```py
# Negative Binomial :
from scipy.stats import nbinom
r=5 # number of successes
p=0.5 # probability of Success
k = range(0,25) # number of failures
# -1 here is the location parameter for generating the PMF we want.
pmf = nbinom.pmf(k, r, p)
chart_creator(k,pmf,"Nbinom PMF")
```

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/output_14_0.png"  height="400" width="700" ></center>
</div>


```py
#Poisson
from scipy.stats import poisson
lamb = .3 # Rate
k = range(0,5)
pmf = poisson.pmf(k, lamb)
chart_creator(k,pmf,"Poisson PMF")
```

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/output_15_0.png"  height="400" width="700" ></center>
</div>

## References:

1. [Introduction to Probability by Joe Blitzstein](http://amzn.to/2xAsYzE)
2. [Wikipedia](https://en.wikipedia.org/wiki/Negative_binomial_distribution)

Next thing I want to come up with is a same sort of post for continuous distributions too. Keep checking for the same. Till then Ciao.
