---
title: "Create basic graph visualizations with SeaBorn- The Most Awesome Python Library For Visualization yet"
date:  2015-09-13
draft: false
url : blog/2015/09/13/seaborn_visualizations/
slug: seaborn_visualizations
Category: Python, Visualization, Statistics
Keywords:
- Python Visualizations
-  Seaborn
-  Matplotlib
-  ggplot2
-  stanford software seaborn
-  regplot
-  lmplot seaborn
-  pairplot seaborn
-  best python visualizations
-  best way to create python visuals
-  python visuals better than ggplot2
-  barplot
-  scatterplot
-  lineplot
-  pairplot
- seaborn scatter_kws
- seaborn set title
- seaborn plot title
- seaborn regplot example
- scatter_kws seaborn
- seaborn regplot title
- seaborn lmplot figure size
- seaborn barplot title
- seaborn title
- seaborn regplot
- seaborn xlabel
- seaborn figure size
- add title to seaborn plot
- seaborn lmplot title
- seaborn figure title
- seaborn histogram
- seaborn jointplot title
- set title seaborn
- seaborn distplot title
- seaborn set figure size
- sns.distplot title
- sns distplot title
- sns regplot
- sns.barplot title
- sns.regplot
- sns.lmplot
- sns figure size
- sns plot title
- sns barplot title
- sns regplot hue
- sns pairplot title
- sns.lmplot figure size
- sns barplot size
- sns regplot color
- sns.regplot example

Tags:
- Visualization
- Python
- Data Science
- Machine Learning
- Best Content

description: This post provides a template to use Seaborn to create customized plots.
toc : false

Categories:
- Data Science
- Awesome Guides

type : post
thumbnail: /images/category_bgs/default_bg.jpg
image: /images/category_bgs/default_bg.jpg
---

When it comes to data preparation and getting acquainted with data, the **one step we normally skip is the data visualization**.
While a part of it could be attributed to the **lack of good visualization tools** for the platforms we use, most of us also **get lazy** at times.

Now as we know of it Python never had any good Visualization library. For most of our plotting needs, I would read up blogs, hack up with StackOverflow solutions and haggle with <a href="http://matplotlib.org/" target="_blank" rel="nofollow">Matplotlib</a> documentation each and every time I needed to make a simple graph. This led me to think that a **Blog post to create common Graph types** in Python is in order. But being the procrastinator that I am it always got pushed to the back of my head.

One thing that helped me in pursuit of my data visualization needs in Python was this awesome course about <a href="https://www.coursera.org/specializations/data-science-python?ranMID=40328&ranEAID=lVarvwc5BD0&ranSiteID=lVarvwc5BD0-SAQTYQNKSERwaOgd07RrHg&siteID=lVarvwc5BD0-SAQTYQNKSERwaOgd07RrHg&utm_content=3&utm_medium=partners&utm_source=linkshare&utm_campaign=lVarvwc5BD0" target="_blank" rel="nofollow">Data Visualization and applied plotting</a> from University of Michigan which is a part of a pretty good <a href="https://www.coursera.org/specializations/data-science-python?ranMID=40328&ranEAID=lVarvwc5BD0&ranSiteID=lVarvwc5BD0-SAQTYQNKSERwaOgd07RrHg&siteID=lVarvwc5BD0-SAQTYQNKSERwaOgd07RrHg&utm_content=3&utm_medium=partners&utm_source=linkshare&utm_campaign=lVarvwc5BD0" target="_blank" rel="nofollow">Data Science Specialization with Python</a> in itself. Highly Recommended.

But, yesterday I got introduced to **<a href="http://stanford.edu/~mwaskom/software/seaborn/" target="_blank" rel="nofollow">Seaborn</a>** and I must say I am **quite impressed** with it. It makes **beautiful graphs** that are in my opinion **better than R's <a href="http://ggplot2.org" target="_blank" rel="nofollow">ggplot2</a>**. Gives you enough options to **customize** and the best part is that it is so **easy to learn**.

So I am finally writing this blog post with a basic **purpose of creating a code base** that provides me with ready to use codes which could be put into analysis in a fairly straight-forward manner.

Right. So here Goes.

We Start by importing the libraries that we will need to use.

```py
import matplotlib.pyplot as plt  #sets up plotting under plt
import seaborn as sns 			#sets up styles and gives us more plotting options
import pandas as pd 			#lets us handle data as dataframes
```

To create a use case for our graphs, we will be working with the **Tips data** that contains the following information.

```py
tips = sns.load_dataset("tips")
tips.head()
```

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/tips.png"  height="400" width="500" ></center>
</div>

## Scatterplot With Regression Line

Now let us work on visualizing this data.
We will use the **<a href="http://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.regplot.html#seaborn.regplot" target="_blank" rel="nofollow">regplot</a>** option in seaborn.

```py
# We dont Probably need the Gridlines. Do we? If yes comment this line
sns.set(style="ticks")

# Here we create a matplotlib axes object. The extra parameters we use
# "ci" to remove confidence interval
# "marker" to have a x as marker.
# "scatter_kws" to provide style info for the points.[s for size]
# "line_kws" to provide style info for the line.[lw for line width]

g = sns.regplot(x="tip", y="total_bill", data=tips, ci = False,
	scatter_kws={"color":"darkred","alpha":0.3,"s":90},
	line_kws={"color":"g","alpha":0.5,"lw":4},marker="x")

# remove the top and right line in graph
sns.despine()

# Set the size of the graph from here
g.figure.set_size_inches(12,8)
# Set the Title of the graph from here
g.axes.set_title('Total Bill vs. Tip', fontsize=34,color="r",alpha=0.5)
# Set the xlabel of the graph from here
g.set_xlabel("Tip",size = 67,color="r",alpha=0.5)
# Set the ylabel of the graph from here
g.set_ylabel("Total Bill",size = 67,color="r",alpha=0.5)
# Set the ticklabel size and color of the graph from here
g.tick_params(labelsize=14,labelcolor="black")
```

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/regplot.png"></center>
</div>


Now that required a bit of a code but i feel that it **looks much better than what either Matplotlib or ggPlot2 could have rendered**. We got a lot of customization without too much code.

But that is not really what actually made me like Seaborn. The plot type that actually got my attention was **<a href="http://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.lmplot.html#seaborn.lmplot" target="_blank" rel="nofollow">lmplot</a>**, which lets us use **regplot** in a **faceted** mode.

```py
# So this function creates a faceted plot. The plot is parameterized by the following:

# col : divides the data points into days and creates that many plots
# palette: deep, muted, pastel, bright, dark, and colorblind. change the colors in graph. Experiment with these
# col_wrap: we want 2 graphs in a row? Yes.We do
# scatter_kws: attributes for points
# hue: Colors on a particular column.
# size: controls the size of graph

g = sns.lmplot(x="tip", y="total_bill",ci=None,data=tips, col="day",
	palette="muted",col_wrap=2,scatter_kws={"s": 100,"alpha":.5},
	line_kws={"lw":4,"alpha":0.5},hue="day",x_jitter=1.0,y_jitter=1.0,size=6)

# remove the top and right line in graph
sns.despine()
# Additional line to adjust some appearance issue
plt.subplots_adjust(top=0.9)

# Set the Title of the graph from here
g.fig.suptitle('Total Bill vs. Tip', fontsize=34,color="r",alpha=0.5)

# Set the xlabel of the graph from here
g.set_xlabels("Tip",size = 50,color="r",alpha=0.5)

# Set the ylabel of the graph from here
g.set_ylabels("Total Bill",size = 50,color="r",alpha=0.5)

# Set the ticklabel size and color of the graph from here
titles = ['Thursday','Friday','Saturday','Sunday']
for ax,title in zip(g.axes.flat,titles):
    ax.tick_params(labelsize=14,labelcolor="black")
```

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/lmplot.png"></center>
</div>

<div style="color:black; background-color: #E9DAEE;">
<a href="http://stanford.edu/~mwaskom/software/seaborn/tutorial/color_palettes.html#building-color-palettes-with-color-palette"  target="_blank" rel="nofollow"><strong>A side Note on Palettes</strong></a>:<br>
You can build your own color palettes using <strong>color_palette()</strong> function.
color_palette() will accept the name of any <strong>seaborn palette</strong> or <a href="http://matplotlib.org/users/colormaps.html"  target="_blank" rel="nofollow"><strong>matplotlib colormap</strong></a>(except jet, which you should never use). It can also take a <strong>list of colors</strong> specified in any valid matplotlib format (RGB tuples, <strong>hex color codes</strong>, or HTML color names).
The return value is always a list of RGB tuples. This allows you to use your own color palettes in graph.
</div>

<script src="//z-na.amazon-adsystem.com/widgets/onejs?MarketPlace=US&adInstanceId=c4ca54df-6d53-4362-92c0-13cb9977639e"></script>

## Barplots

```py
sns.set(style="ticks")

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

# This Function takes as input a custom palette
g = sns.barplot(x="sex", y="tip", hue="day",
	palette=sns.color_palette(flatui),data=tips,ci=None)

# remove the top and right line in graph
sns.despine()

# Set the size of the graph from here
g.figure.set_size_inches(12,7)
# Set the Title of the graph from here
g.axes.set_title('Do We tend to \nTip high on Weekends?',
	fontsize=34,color="b",alpha=0.3)
# Set the xlabel of the graph from here
g.set_xlabel("Gender",size = 67,color="g",alpha=0.5)
# Set the ylabel of the graph from here
g.set_ylabel("Mean Tips",size = 67,color="r",alpha=0.5)
# Set the ticklabel size and color of the graph from here
g.tick_params(labelsize=14,labelcolor="black")
```

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/barplot.png"></center>
</div>

## Histograms and Distribution Diagrams

They form another part of my workflow. Lets plot the normal Histogram using seaborn.
For this we will use the **<a href="http://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.distplot.html#seaborn.distplot" target="_blank" rel="nofollow">distplot</a>** function. This function combines the matplotlib hist function (with automatic calculation of a good default bin size) with the seaborn kdeplot() function.
It can also fit **scipy.stats** distributions and plot the estimated PDF over the data.

```py
# Create a list of 1000 Normal RVs
x = np.random.normal(size=1000)

sns.set_context("poster")
sns.set_style("ticks")
# This  Function creates a normed Histogram by default.
# If we use the parameter kde=False and norm_hist=False then
# we will be using a count histogram

g=sns.distplot(x,
         	kde_kws={"color":"g","lw":4,"label":"KDE Estim","alpha":0.5},
            hist_kws={"color":"r","alpha":0.3,"label":"Freq"})


# remove the top and right line in graph
sns.despine()

# Set the size of the graph from here
g.figure.set_size_inches(12,7)
# Set the Title of the graph from here
g.axes.set_title('Normal Simulation', fontsize=34,color="b",alpha=0.3)
# Set the xlabel of the graph from here
g.set_xlabel("X",size = 67,color="g",alpha=0.5)
# Set the ylabel of the graph from here
g.set_ylabel("Density",size = 67,color="r",alpha=0.5)
# Set the ticklabel size and color of the graph from here
g.tick_params(labelsize=14,labelcolor="black")
```

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/hist_normal.png"></center>
</div>

```py
import scipy.stats as stats

a = 1.5
b = 1.5
x = np.arange(0.01, 1, 0.01)
y = stats.beta.rvs(a,b,size=10000)
y_act = stats.beta.pdf(x,a,b)
g=sns.distplot(y,kde=False,norm_hist=True,
            kde_kws={"color":"g","lw":4,"label":"KDE Estim","alpha":0.5},
            hist_kws={"color":"r","alpha":0.3,"label":"Freq"})
# Note that we plotted on the graph using plt matlabplot function
plt.plot(x,y_act)

# remove the top and right line in graph
sns.despine()

# Set the size of the graph from here
g.figure.set_size_inches(12,7)
# Set the Title of the graph from here
g.axes.set_title(("Beta Simulation vs. Calculated Beta Density\nFor a=%s,b=%s")
	%(a,b),fontsize=34,color="b",alpha=0.3)
# Set the xlabel of the graph from here
g.set_xlabel("X",size = 67,color="g",alpha=0.5)
# Set the ylabel of the graph from here
g.set_ylabel("Density",size = 67,color="r",alpha=0.5)
# Set the ticklabel size and color of the graph from here
g.tick_params(labelsize=14,labelcolor="black")
```

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/hist_beta.png"></center>
</div>

## PairPlots
You need to see how variables vary with one another. What is the distribution of variables in the dataset. This is the graph to use with the **<a href="http://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.pairplot.html#seaborn.pairplot" target="_blank" rel="nofollow">pairplot</a>** function. Very helpful And Seaborn males it a joy to use. We will use **Iris Dataset** here for this example.

```py
iris = sns.load_dataset("iris")
iris.head()
```

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/iris.png" height="500" width="600"></center>
</div>

```py
# Create a Pairplot
g = sns.pairplot(iris,hue="species",palette="muted",size=5,
	vars=["sepal_width", "sepal_length"],kind='reg',markers=['o','x','+'])

# To change the size of the scatterpoints in graph
g = g.map_offdiag(plt.scatter,  s=35,alpha=0.5)

# remove the top and right line in graph
sns.despine()
# Additional line to adjust some appearance issue
plt.subplots_adjust(top=0.9)

# Set the Title of the graph from here
g.fig.suptitle('Relation between Sepal Width and Sepal Length',
	fontsize=34,color="b",alpha=0.3)
```

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/pairplot.png"></center>
</div>


Hope you found this post useful and worth your time. You can find the iPython notebook at <a href="https://github.com/MLWhiz/visualization/blob/master/Graphs.ipynb" target="_blank" rel="nofollow">github</a>

I tried to make this as simple as possible but You may always **ask me** or see the documentation for doubts.

If you have **any more ideas** on how to use Seaborn or **which graphs should i add here**, please suggest in the **comments** section.

I will definitely try to add to this post as I start using more visualizations and encounter other libraries as good as seaborn.

Also since this is my first visualization post on this blog, I would like to call out a good course about <a href="https://www.coursera.org/specializations/data-science-python?ranMID=40328&ranEAID=lVarvwc5BD0&ranSiteID=lVarvwc5BD0-SAQTYQNKSERwaOgd07RrHg&siteID=lVarvwc5BD0-SAQTYQNKSERwaOgd07RrHg&utm_content=3&utm_medium=partners&utm_source=linkshare&utm_campaign=lVarvwc5BD0" target="_blank" rel="nofollow">Data Visualization and applied plotting</a> from University of Michigan which is a part of a pretty good <a href="https://www.coursera.org/specializations/data-science-python?ranMID=40328&ranEAID=lVarvwc5BD0&ranSiteID=lVarvwc5BD0-SAQTYQNKSERwaOgd07RrHg&siteID=lVarvwc5BD0-SAQTYQNKSERwaOgd07RrHg&utm_content=3&utm_medium=partners&utm_source=linkshare&utm_campaign=lVarvwc5BD0" target="_blank" rel="nofollow">Data Science Specialization with Python</a> in itself. Do check it out.
