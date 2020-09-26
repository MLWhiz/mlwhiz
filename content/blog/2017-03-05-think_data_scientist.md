---
title: "Top advice for a Data Scientist"
date:  2017-03-05
draft: false
url : blog/2017/03/05/think_like_a_data_scientist/
slug: think_like_a_data_scientist
Category: Data Science, Statistics, Resources, Learning, Books, Python, Distributions, Statistical Inference, hadoop, spark, deep learning
Keywords:
- data scientist
-  think
- pitfalls
-  Learn Deep Learning
-  Learn Statistics
-  Learn Data science
-  Learn Big Data
-  Learn Bash
-  Learn Linux
-  Learn Spark
-  Learn Linear Algebra
- Data Science
-  Statistics
-  Resources
-  Learning
-  Books
-  Python
-  Distributions
-  Statistical Inference
-  hadoop
-  spark
-  deep learning
-  learn hadoop
-  top data science resources
-  top machine learning resources
-  top ai resources
-  top learning resources for data science
-  best resources for data science
-  great resources for data science
-  awesome resources for data science
- Top Advice for a data scientist
- Pitfalls and mistakes data science
-  think like data scientist
- Python
- data scientist
- algorithms

Tags:
- Data Science
- Opinion


description:  How to think like a Data Scientist. Pitfalls and mistakes young data scientists make. Top Advice for a data scientist.
toc : false

Categories:
- Data Science
type : post
thumbnail: /images/category_bgs/default_bg.jpg
image: /images/category_bgs/default_bg.jpg
---



A data scientist needs to be Critical and always on a lookout of something that misses others. So here are some advices that one can include in day to day data science work to be better at their work:

## 1. Beware of the Clean Data Syndrome

You need to ask yourself questions even before you start working on the data. **Does this data make sense?** Falsely assuming that the data is clean could lead you towards wrong Hypotheses. Apart from that, you can discern a lot of important patterns by looking at discrepancies in the data. For example, if you notice that a particular column has more than 50% values missing, you might think about not using the column. Or you may think that some of the data collection instrument has some error.

Or let's say you have a distribution of Male vs Female as 90:10 in a Female Cosmetic business. You may assume clean data and show the results as it is or you can use common sense and ask if the labels are switched.

## 2. Manage Outliers wisely

Outliers can help you understand more about the people who are using your website/product 24 hours a day. But including them while building models will skew the models a lot.

## 3. Keep an eye out for the Abnormal

Be on the **lookout for something out of the obvious**. If you find something you may have hit gold.

For example, [Flickr started up as a Multiplayer game](https://www.fastcompany.com/1783127/flickr-founders-glitch-can-game-wants-you-play-nice-be-blockbuster). Only when the founders noticed that people were using it as a photo upload service, did they pivot.

Another example: fab.com started up as fabulis.com, a site to help gay men meet people. One of the site's popular features was the "Gay deal of the Day". One day the deal was for Hamburgers - and half of the buyers were women. This caused the team to realize that there was a market for selling goods to women. So Fabulis pivoted to fab as a flash sale site for designer products.

## 4. Start Focussing on the right metrics

- **Beware of Vanity metrics** For example, # of active users by itself doesn't divulge a lot of information. I would rather say "5% MoM increase in active users" rather than saying " 10000 active users". Even that is a vanity metric as active users would always increase. I would rather keep a track of percentage of users that are active to know how my product is performing.
- Try to find out a **metric that ties with the business goal**. For example, Average Sales/User for a particular month.

<script src="//z-na.amazon-adsystem.com/widgets/onejs?MarketPlace=US&adInstanceId=c4ca54df-6d53-4362-92c0-13cb9977639e"></script>

## 5. Statistics may lie too

Be critical of everything that gets quoted to you. Statistics has been used to lie in advertisements, in workplaces and a lot of other marketing venues in the past. People will do anything to get sales or promotions.

For example: [Do you remember Colgate’s claim that 80% of dentists recommended their brand?](http://marketinglaw.osborneclarke.com/retailing/colgates-80-of-dentists-recommend-claim-under-fire/)

This statistic seems pretty good at first. It turns out that at the time of surveying the dentists, they could choose several brands — not just one. So other brands could be just as popular as Colgate.

Another Example: **"99 percent Accurate" doesn't mean shit**. Ask me to create a cancer prediction model and I could give you a 99 percent accurate model in a single line of code. How? Just predict "No Cancer" for each one. I will be accurate may be more than 99% of the time as Cancer is a pretty rare disease. Yet I have achieved nothing.

## 6. Understand how probability works

It happened during the summer of 1913 in a Casino in Monaco. Gamblers watched in amazement as a casino's roulette wheel landed on black 26 times in a row. And since the probability of a Red vs Black is exactly half, they were certain that red was "due". It was a field day for the Casino. A perfect example of [Gambler's fallacy](https://en.wikipedia.org/wiki/Gambler's_fallacy), aka the Monte Carlo fallacy.

And This happens in real life. [People tend to avoid long strings of the same answer](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2538147). Sometimes sacrificing accuracy of judgment for the sake of getting a pattern of decisions that looks fairer or probable.

For example, An admissions officer may reject the next application if he has approved three applications in a row, even if the application should have been accepted on merit.

## 7. Correlation Does Not Equal Causation

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/corr_caus.png"  height="400" width="500" ></center>
</div>

The Holy Grail of a Data scientist toolbox. To see something for what it is. Just because two variables move together in tandem doesn't necessarily mean that one causes the another. There have been hilarious examples for this in the past. Some of my favorites are:

- Looking at the firehouse department data you infer that the more firemen are sent to a fire, the more damage is done.

- When investigating the cause of crime in New York City in the 80s, an academic found a strong correlation between the amount of serious crime committed and the amount of ice cream sold by street vendors! Obviously, there was an unobserved variable causing both. Summers are when the crime is the greatest and when the most ice cream is sold. So Ice cream sales don't cause crime. Neither crime increases ice cream sales.

## 8. More data may help

Sometimes getting extra data may work wonders. You might be able to model the real world more closely by looking at the problem from all angles. Look for extra data sources.

For example, Crime data in a city might help banks provide a better credit line to a person living in a troubled neighborhood and in turn increase the bottom line.
