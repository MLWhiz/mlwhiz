---
title: "Maths Beats Intuition probably every damn time"
date:  2017-04-16
draft: false
url : blog/2017/04/16/maths_beats_intuition/
slug: maths_beats_intuition
Category: Python, machine learning, probability
Keywords: 
- birthday problem
- birthday problem simulation python
Tags: 
- Python
- probability
- Data Science
- Statistics
description:  A Simulation + intuition based approach to solve the birthday problem.
toc : false
---

Newton once said that **"God does not play dice with the universe"**. But actually he does. Everything happening around us could be explained in terms of probabilities. We repeatedly watch things around us happen due to chances, yet we never learn. We always get dumbfounded by the playfulness of nature.

One of such ways intuition plays with us is with the Birthday problem.

## Problem Statement:

*In a room full of N people, what is the probability that 2 or more people share the same birthday(Assumption: 365 days in year)?*

By the [pigeonhole principle](https://en.wikipedia.org/wiki/Pigeonhole_principle), the probability reaches 100% when the number of people reaches 366 (since there are only 365 possible birthdays).

**However, the paradox is that 99.9% probability is reached with just 70 people, and 50% probability is reached with just 23 people.**


## Mathematical Proof:

Sometimes a good strategy when trying to find out probability of an event is to look at the probability of the complement event.Here it is easier to find the probability of the complement event.
We just need to count the number of cases in which no person has the same birthday.(Sampling without replacement)
Since there are k ways in which birthdays can be chosen with replacement.

$P(birthday Match) = 1 - \dfrac{(365).364...(365−k+1)}{365^k}$

## Simulation:

Lets try to build around this result some more by trying to simulate this result:

```py
%matplotlib inline
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt  #sets up plotting under plt
import seaborn as sns           #sets up styles and gives us more plotting options
import pandas as pd             #lets us handle data as dataframes
import random

def sim_bithday_problem(num_people_room, trials =1000):
    '''This function takes as input the number of people in the room.
    Runs 1000 trials by default and returns
    (number of times same brthday found)/(no of trials)
    '''
    same_birthdays_found = 0
    for i in range(trials):
        # randomly sample from the birthday space which could be any of a number from 1 to 365
        birthdays = [random.randint(1,365) for x in range(num_people_room)]
        if len(birthdays) - len(set(birthdays))>0:
            same_birthdays_found+=1
    return same_birthdays_found/float(trials)

num_people = range(2,100)
probs = [sim_bithday_problem(i) for i in num_people]
data = pd.DataFrame()
data['num_peeps'] = num_people
data['probs'] = probs
sns.set(style="ticks")

g = sns.regplot(x="num_peeps", y="probs", data=data, ci = False,
    scatter_kws={"color":"darkred","alpha":0.3,"s":90},
    marker="x",fit_reg=False)

sns.despine()
g.figure.set_size_inches(10,6)
g.axes.set_title('As the Number of people in room reaches 23 the probability reaches ~0.5\nAt more than 50 people the probability is reaching 1', fontsize=15,color="g",alpha=0.5)
g.set_xlabel("# of people in room",size = 30,color="r",alpha=0.5)
g.set_ylabel("Probability",size = 30,color="r",alpha=0.5)
g.tick_params(labelsize=14,labelcolor="black")
```

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/bithdayproblem.png"  height="400" width="700" ></center>
</div>

We can see from the [graph](/blog/2015/09/13/seaborn_visualizations/) that as the Number of people in room reaches 23 the probability reaches ~ 0.5. So we have proved this fact Mathematically as well as with simulation.

## Intuition:

To understand it we need to think of this problem in terms of pairs. There are ${{23}\choose{2}} = 253$ pairs of people in the room when only 23 people are present. Now with that big number you should not find the probability of 0.5 too much. In the case of 70 people we are looking at ${{70}\choose{2}} = 2450$ pairs.

So thats it for now. To learn more about this go to [Wikipedia](https://en.wikipedia.org/wiki/Birthday_problem) which has an awesome page on this topic.

## References:

1. [Introduction to Probability by Joseph K. Blitzstein](http://amzn.to/2nIUkxq)
2. [Birthday Problem on Wikipedia](https://en.wikipedia.org/wiki/Birthday_problem)
