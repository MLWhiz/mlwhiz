---
title: "Python’s One Liner graph creation library with animations Hans Rosling Style"
date:  2019-05-05
draft: false
url : blog/2019/05/05/plotly_express/
slug: plotly_express

Categories:
- Data Science

Keywords:
- Python Visualizations

Tags:
- Visualization
- Python
- Machine Learning
- Data Science

description: Some awesome visualizations using Plotly Express. Create Hans Rosling animations with a single line of code.

thumbnail : /images/plotly_ex/visualization.png
image : /images/plotly_ex/visualization.png
toc : false
type: "post"
---



I distinctly remember the time when Seaborn came. I was really so fed up with Matplotlib. To create even simple graphs I had to run through so many StackOverflow threads.

***The time I could have spent in thinking good ideas for presenting my data was being spent in handling Matplotlib. And it was frustrating.***

Seaborn is much better than Matplotlib, yet it also demands a lot of code for a simple “good looking” graph.

When Plotly came it tried to solve that problem. And when added with Pandas, plotly is a great tool.

Just using the iplot function, you can do so much with Plotly.

***But still, it is not very intuitive. At least not for me.***

I still didn’t switch to Plotly just because I had spent enough time with Seaborn to do things “quickly” enough and I didn’t want to spend any more time learning a new visualization library. I had created [my own functions](https://towardsdatascience.com/3-awesome-visualization-techniques-for-every-dataset-9737eecacbe8?source=friends_link&sk=e32fe5bcd7d6553fca6d7f371980089f) in Seaborn to create the visualizations I most needed. ***Yet it was still a workaround. I had given up hope of having anything better.***

Comes ***Plotly Express*** in the picture. And is it awesome?

According to the creators of Plotly Express (who also created Plotly obviously), ***Plotly Express is to Plotly what Seaborn is to Matplotlib.***
> # A terse, consistent, high-level wrapper around Plotly.py for rapid data exploration and figure generation.

I just had to try it out.

And have the creators made it easy to start experimenting with it?

***One-liners to do everything you want?*** ✅

***Standardized functions?*** Learn to create a scatterplot and you have pretty much learned this tool — ✅

**Interactive graphs?** ✅

**Animations?** Racing Bar plots, Scatter plots with time, Maps ✅

**Free and Open Source?** ✅

Just a sneak peek of what we will be able to create(and more) by the end of this post. ***Using a single line of code.***

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/plotly_ex/animated1.gif" "></center>
</div>

Ok enough of the talk, let’s get to it.

## First the Dataset — Interesting, Depressing and Inspiring all at once

![](https://cdn-images-1.medium.com/max/3600/1*VL1Vl8hJk7Kfka5atIzH8Q.png)

We will be working with the [Suicide dataset](https://www.kaggle.com/russellyates88/suicide-rates-overview-1985-to-2016) I took from Kaggle. This dataset is compiled from data taken from the UN, World Bank and World Health Organization. ***The dataset was accumulated with the inspiration for Suicide Prevention. I am always up for such good use of data.***

You can find all the code for this post and run it yourself in this [Kaggle Kernel](https://www.kaggle.com/mlwhiz/plotly-express/)

First I will do some data Cleaning to add continent information and Country ISO codes as they will be helpful later:

```py
import pandas as pd
import numpy as np
import plotly_express as px

# Suicide Data
suicides = pd.read_csv("../input/suicide-rates-overview-1985-to-2016/master.csv")
del suicides['HDI for year']
del suicides['country-year']

# Country ISO Codes
iso_country_map = pd.read_csv("../input/countries-iso-codes/wikipedia-iso-country-codes.csv")
iso_country_map = iso_country_map.rename(columns = {'English short name lower case':"country"})

# Load Country Continents file
concap =pd.read_csv("../input/country-to-continent/countryContinent.csv", encoding='iso-8859-1')[['code_3', 'continent', 'sub_region']]
concap = concap.rename(columns = {'code_3':"Alpha-3 code"})

correct_names = {'Cabo Verde': 'Cape Verde', 'Macau': 'Macao', 'Republic of Korea': "Korea, Democratic People's Republic of" ,
 'Russian Federation': 'Russia',
 'Saint Vincent and Grenadines':'Saint Vincent and the Grenadines'
 , 'United States': 'United States Of America'}

def correct_country(x):
    if x in correct_names:
        return correct_names[x]
    else:
        return x

suicides['country'] = suicides['country'].apply(lambda x : correct_country(x))

suicides = pd.merge(suicides,iso_country_map,on='country',how='left')
suicides = pd.merge(suicides,concap,on='Alpha-3 code',how='left')

suicides['gdp'] = suicides['gdp_per_capita ($)']*suicides['population']
```
Let us look at the suicides data:

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/plotly_ex/data_img.png" "></center>
</div>


I will also group the data by continents. Honestly, I am doing this only to show the power of the library as the main objective of this post will still be to create awesome visualizations.

```py
suicides_gby_Continent = suicides.groupby(['continent','sex','year']).aggregate(np.sum).reset_index()

suicides_gby_Continent['gdp_per_capita ($)'] = suicides_gby_Continent['gdp']/suicides_gby_Continent['population']

suicides_gby_Continent['suicides/100k pop'] = suicides_gby_Continent['suicides_no']*1000/suicides_gby_Continent['population']

# 2016 data is not full
suicides_gby_Continent=suicides_gby_Continent[suicides_gby_Continent['year']!=2016]

suicides_gby_Continent.head()
```

The final data we created:

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/plotly_ex/data_2.png" "></center>
</div>

## Simplicity of use

We are ready to visualize our data. Comes ***Plotly Express*** time. I can install it just by a simple:
```bash
pip install plotly_express
```
and import it as:

```py
import plotly_express as px
```
Now let us create a simple scatter plot with it.
```py
suicides_gby_Continent_2007 = suicides_gby_Continent[suicides_gby_Continent['year']==2007]

px.scatter(suicides_gby_Continent_2007,x = 'suicides/100k pop', y = 'gdp_per_capita ($)')
```

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/plotly_ex/point_plot.png" "></center>
</div>

Not very inspiring. Right. Let us make it better step by step. Lets color the points by Continent.

```py
px.scatter(suicides_gby_Continent_2007,x = 'suicides/100k pop', y = 'gdp_per_capita ($)',color='continent')
```

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/plotly_ex/colored_point_plot.png" "></center>
</div>

**Better but not inspiring. YET.**

**The points look so small**. Right. Let us increase the point size. How? What could the parameter be….

```py
px.scatter(suicides_gby_Continent_2007,x = 'suicides/100k pop', y = 'gdp_per_capita ($)',color='ContinentName',size ='suicides/100k pop')
```

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/plotly_ex/sized_point_plot.png" "></center>
</div>


**Can you see there are two points for every continent?** They are for male and female. Let me show that in the graph. We can show this distinction using a couple of ways. ***We can use different symbol or use different facets for male and female.***

Let me show them both.
```py
px.scatter(suicides_gby_Continent_2007,x = 'suicides/100k pop', y = 'gdp_per_capita ($)', size = 'suicides/100k pop', color='ContinentName',symbol='sex')
```
<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/plotly_ex/facet1.png" "></center>
</div>

![](https://cdn-images-1.medium.com/max/2000/1*H12_tRVLrX8znkb7WWicvw.png)

We could also create a faceted plot.

```py
px.scatter(suicides_gby_Continent_2007,x = 'suicides/100k pop', y = 'gdp_per_capita ($)', size = 'suicides/100k pop', color='continent',facet_col='sex')
```
<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/plotly_ex/facet2.png" "></center>
</div>

The triangles are for male and the circles are for females in the symbol chart. ***We are already starting to see some good info from the chart.*** For example:

* There is a significant difference between the suicide rates of Male vs Females at least in 2007 data.

* ***European Males were highly susceptible to Suicide in 2007?***

* The income disparity doesn’t seem to play a big role in suicide rates. Asia has a lower GDP per capita and a lower suicide rate than Europe.

* ***There doesn’t seem to be income disparity amongst males and females.***

Still not inspiring? Umm. ***Let us add some animation***. That shouldn’t have to be hard. I will just add some more parameters,

* animation_frame which specifies what will be our animation dimension.

* range of x and y values using range_y and range_x

* text which labels all points with continents. Helps in visualizing data better

```py
px.scatter(suicides_gby_Continent,x = 'suicides/100k pop', y = 'gdp_per_capita ($)',color='continent',
               size='suicides/100k pop',symbol='sex',animation_frame='year', animation_group='continent',range_x = [0,0.6],
              range_y = [0,70000],text='continent')
```

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/plotly_ex/animated1.gif" "></center>
</div>



**Wait for the gif plot to show.**

In the Jupyter notebook, you will be able to stop the visualization, hover over the points, just look at a particular continent and do so much more with interactions.

So much information with a single command. We can see that:

* From 1991–2001 European Males had a pretty bad Suicide rate.

* Oceania even after having a pretty high GDP per capita, it is still susceptible to suicides.

* Africa has lower suicide rates as compared to other countries.

* For the Americas, the suicide rates have been increasing gradually.

***All of my above observations would warrant more analysis. But that is the point of having so much information on a single graph. It will help you to come up with a lot of hypotheses.***

The above style of the plot is known as **Hans Rosling plot** named after its founder.

Here I would ask you to see this presentation from Hans Rosling where he uses Gapminder data to explain how income and lifespan emerged in the world through years. See it. It's great.

<center><iframe width="560" height="315" src="https://www.youtube.com/embed/jbkSRLYSojo" frameborder="0" allowfullscreen></iframe></center>

## Function Standardization

***So till now, we have learned about scatter plots. So much time to just learn a single class of charts.*** In the start of my post, I told you that this library has a sort of standardized functions.

Let us specifically look at European data as we saw that European males have a high Suicide rate.

```py
european_suicide_data = suicides[suicides['continent'] =='Europe']
european_suicide_data_gby = european_suicide_data.groupby(['age','sex','year']).aggregate(np.sum).reset_index()
european_suicide_data_gby['suicides/100k pop'] = european_suicide_data_gby['suicides_no']*1000/european_suicide_data_gby['population']


# A single line to create an animated Bar chart too.
px.bar(european_suicide_data_gby,x='age',y='suicides/100k pop',facet_col='sex',animation_frame='year',
       animation_group='age',
       category_orders={'age':['5-14 years', '15-24 years', '25-34 years', '35-54 years',
       '55-74 years', '75+ years']},range_y=[0,1])
```

![](https://cdn-images-1.medium.com/max/2328/1*F1AvmKFQdWB3uswpj2Au_g.gif)

Just like that, we have learned about animating our bar plots too. In the function above I provide a category_order for the axes to force the order of categories since they are ordinal. Rest all is still the same.

*We can see that from 1991 to 2001 the suicide rate of 75+ males was very high. That might have increased the overall suicide rate for males.*

***Want to see how the suicide rates decrease in a country using a map?*** That is why we got the ISO-codes for the country in the data.

How many lines should that take? You guessed right. One.

```py
suicides_map = suicides.groupby(['year','country','Alpha-3 code']).aggregate(np.sum).reset_index()[['country','Alpha-3 code','suicides_no','population','year']]

suicides_map["suicides/100k pop"]=suicides_map["suicides_no"]*1000/suicides_map["population"]

px.choropleth(suicides_map, locations="Alpha-3 code", color="suicides/100k pop", hover_name="country", animation_frame="year",
             color_continuous_scale=px.colors.sequential.Plasma)
```
![](https://cdn-images-1.medium.com/max/2328/1*YOKJJFQGWnZ9QDb_TjRY-Q.gif)

The plot above shows how suicide rates have changed over time in different countries and based on the info we get from the plot the coding effort required is minimal. We can see that:

* A lot of countries are missing

* Africa has very few countries in data

* Almost all of Asia is also missing.

We can get quite a good understanding of our data just by seeing the above graphs.

***Animations on the time axis also add up a lot of value as we are able to see all our data using a single graph.***

This can help us in finding hidden patterns in the data. And you have to agree, it looks cool too.

## Conclusion

This was just a preview of Plotly Express. You can do a lot of other things using this library.

The main thing I liked about this library is the way it has tried to simplify graph creation. And how the graphs look cool out of the box.

***Just think of the lengths one would have to go to to create the same graphs in Seaborn or Matplotlib or even Plotly***. And you will be able to appreciate the power the library provides even more.

There is a bit of lack of documentation for this project by Plotly, but I found that the functions are pretty much well documented. On that note, you can see function definitions using Shift+Tab in Jupyter.

Also as per its announcement article: “Plotly Express is *totally free*: with its permissive open-source MIT license, you can use it however you like (yes, even in commercial products!).”

So there is no excuse left now to put off that visual. Just get to it…

You can find all the code for this post and run it yourself in this [Kaggle Kernel](https://www.kaggle.com/mlwhiz/plotly-express/)

If you want to learn about best strategies for creating Visualizations, I would like to call out an excellent course about [**Data Visualization and applied plotting**](https://imp.i384100.net/JKKJZa) from the University of Michigan which is a part of a pretty good [**Data Science Specialization with Python**](https://imp.i384100.net/JKKJZa) in itself. Do check it out

I am going to be writing more beginner friendly posts in the future too. Follow me up at [**Medium**](https://mlwhiz.medium.com/) or Subscribe to my [**blog**](https://mlwhiz.com/) to be informed about them. As always, I welcome feedback and constructive criticism and can be reached on Twitter [@mlwhiz](https://twitter.com/MLWhiz)
