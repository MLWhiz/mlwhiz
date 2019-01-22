---
title: "Data Science 101 : Playing with Scraping in Python"
date:  2014-10-02
draft: false
url : blog/2014/10/02/data_science_cs_109_download_rtmdump/
slug: data_science_cs_109_download_rtmdump
Category: Python
Keywords: 
- Web Scraping
Tags: 
- Web Scraping
description: This is a simple illustration of using Pattern Module to scrape web data using Python. We will be scraping the data from imdb for the top TV Series along with their ratings
toc : false
---

This is a simple illustration of using Pattern Module to scrape web data using Python. We will be scraping the data from imdb for the top TV Series along with their ratings

We will be using this link for this:

<pre style="font-family:courier new,monospace; background-color:#f6c6529c; color:#000000">
http://www.imdb.com/search/title?count=100&num_votes=5000,&ref_=gnr_tv_hr&sort=user_rating,desc&start=1&title_type=tv_series,mini_series
</pre>

This URL gives a list of top Rated TV Series which have number of votes atleast 5000. The Thing to note in this URL is the "&start=" parameter where we can specify which review should the list begin with. If we specify 1 we will get reviews starting from 1-100, if we specify 101 we get reviews from 101-200 and so on.

Lets Start by importing some Python Modules that will be needed for Scraping Data:

``` py
import requests                     # This is a module that is used for getting html data from a webpage in the text format
from pattern import web             # We use this module to parse through the dtaa that we loaded using requests
```

## Loading the data using requests and pattern
So the modules are loaded at this point, next we will try to catch the url using python and put this into a dict in python. We will start with a single URL and then try to parse it using pattern module


```py
url= "http://www.imdb.com/search/title?count=100&num_votes=5000,&ref_=gnr_tv_hr&sort=user_rating,desc&start=1&title_type=tv_series,mini_series"
html_data = requests.get(url).text 
dom=web.Element(html_data)
```

## Parsing the data
This is the data of Interest found out after some nspection of the html code. This is for a single TV Series Band of brothers, but if you are able to parse this you just have to move hrough a loop.

``` html
<html>
<td class="title">
<span class="wlb_wrapper" data-tconst="tt0185906" data-size="small" data-caller-name="search"></span>
<a href="/title/tt0185906/">Band of Brothers</a>
<span class="year_type">(2001 Mini-Series)</span><br />
<div class="user_rating">
<div class="rating rating-list" data-auth="BCYm-Mk2Ros7BTxsLNL2XJX_icfZVahNr1bE9-5Ajb2N3381yxcaNN4ZQqyrX7KgEFGqHWmwv10lv7lAnXyC8CCkh9hPqQfzwVTumCeRzjpnndW4_ft97qQkBYLUvFxYnFgR" id="tt0185906|imdb|9.6|9.6|advsearch" data-ga-identifier="advsearch" title="Users rated this 9.6/10 (156,073 votes) - click stars to rate">
<span class="rating-bg">&nbsp;</span>
<span class="rating-imdb" style="width: 134px">&nbsp;</span>
<span class="rating-stars">
<a href="/register/login?why=vote" title="Register or login to rate this title" rel="nofollow"><span>1</span></a>
<a href="/register/login?why=vote" title="Register or login to rate this title" rel="nofollow"><span>2</span></a>
<a href="/register/login?why=vote" title="Register or login to rate this title" rel="nofollow"><span>3</span></a>
<a href="/register/login?why=vote" title="Register or login to rate this title" rel="nofollow"><span>4</span></a>
<a href="/register/login?why=vote" title="Register or login to rate this title" rel="nofollow"><span>5</span></a>
<a href="/register/login?why=vote" title="Register or login to rate this title" rel="nofollow"><span>6</span></a>
<a href="/register/login?why=vote" title="Register or login to rate this title" rel="nofollow"><span>7</span></a>
<a href="/register/login?why=vote" title="Register or login to rate this title" rel="nofollow"><span>8</span></a>
<a href="/register/login?why=vote" title="Register or login to rate this title" rel="nofollow"><span>9</span></a>
<a href="/register/login?why=vote" title="Register or login to rate this title" rel="nofollow"><span>10</span></a>
</span>
<span class="rating-rating"><span class="value">9.6</span><span class="grey">/</span><span class="grey">10</span></span>
<span class="rating-cancel"><a href="/title/tt0185906/vote?v=X;k=BCYm-Mk2Ros7BTxsLNL2XJX_icfZVahNr1bE9-5Ajb2N3381yxcaNN4ZQqyrX7KgEFGqHWmwv10lv7lAnXyC8CCkh9hPqQfzwVTumCeRzjpnndW4_ft97qQkBYLUvFxYnFgR" title="Delete" rel="nofollow"><span>X</span></a></span>
&nbsp;</div>
</div>
<span class="outline">The story of Easy Company of the US Army 101st Airborne division and their mission in WWII Europe from Operation Overlord through V-J Day.</span>
<span class="credit">
    With: <a href="/name/nm0342241/">Scott Grimes</a>, <a href="/name/nm0500614/">Matthew Leitch</a>, <a href="/name/nm0507073/">Damian Lewis</a>
</span>
<span class="genre"><a href="/genre/action">Action</a> | <a href="/genre/drama">Drama</a> | <a href="/genre/history">History</a> | <a href="/genre/war">War</a></span>
<span class="certificate"><span title="TV_MA" class="us_tv_ma titlePageSprite"></span></span>
<span class="runtime">705 mins.</span>
</td>
```

Now we have loaded the data we need to parse it using the functions from pattern module.
The main function in pattern module is the by_tag() function which lets you get all the elements with that particular tagname.
For us the main interest is this "td" tag with class as "title". This "td" tag contains:

1. Title in the "a" tag
2. Rating in the "span" tag with class "value"
3. Genres in the "span" tag with class "genre" and then looping through the "a" tags 
4. Runtime in "span" tag with class "runtime"
5. Artists in "span" tag with class "credit" loop through "a" tags

Now lets write some code to parse this data.

```py
for tv_series in dom.by_tag('td.title'):    
    title = tv_series.by_tag('a')[0].content
    genres = tv_series.by_tag('span.genre')[0].by_tag('a')
    genres = [g.content for g in genres]
    try:
        runtime = tv_series.by_tag('span.runtime')[0].content
    except:
        runtime = "NA"
    rating = tv_series.by_tag('span.value')[0].content
    artists = tv_series.by_tag('span.credit')[0].by_tag('a')
    artists = [a.content for a in artists]
    print title, genres, runtime, rating, artists

```

<pre style="font-family:courier new,monospace; background-color:#f6c6529c; color:#000000">Band of Brothers [u'Action', u'Drama', u'History', u'War'] 705 mins. 9.6 [u'Scott Grimes', u'Matthew Leitch', u'Damian Lewis']

Breaking Bad [u'Crime', u'Drama', u'Thriller'] 45 mins. 9.6 [u'Bryan Cranston', u'Aaron Paul', u'Anna Gunn']

Game of Thrones [u'Adventure', u'Drama', u'Fantasy'] 55 mins. 9.5 [u'Lena Headey', u'Peter Dinklage', u'Maisie Williams']</pre>

So finally we are OK with parsing. We have understood the structure of the webpage, the tags and classes we will need to use and how to use pattern module to find data for a single page. Now lets use the power of for loops to get all the data.

### Getting Whole Data

Lets Go through it the pythonic way. We will create functions and try to execute small chunks of code rather than doing it all at once. 
Lets first create a funcion that takes a start_val(for the start parameter) and returns a dom element.

```py
def get_dom(start_val):
    url= "http://www.imdb.com/search/title?count=100&num_votes=5000,&ref_=gnr_tv_hr&sort=user_rating,desc&start="+str(start_val)+"&title_type=tv_series,mini_series"
    html_data = requests.get(url).text 
    dom=web.Element(html_data)
    return dom
```


Now lets create a function parse_dom that takes as input dom an throws out a list containing all the data. The list is like this :
<pre style="font-family:courier new,monospace; background-color:#f6c6529c; color:#000000">
[
['Band of Brothers','Action|Drama|History|War','705 mins.','9.6','Scott Grimes|Matthew Leitch|Damian Lewis'],
['Breaking Bad','Crime|Drama|Thriller','45 mins.', '9.6' ,'Bryan Cranston|Aaron Paul|Anna Gunn'],.....
]	
</pre>

```py
def parse_dom(dom):
    result=[]
    for tv_series in dom.by_tag('td.title'):    
        title = tv_series.by_tag('a')[0].content
        genres = tv_series.by_tag('span.genre')[0].by_tag('a')
        genres = "|".join([g.content for g in genres])
        try:
            runtime = tv_series.by_tag('span.runtime')[0].content
        except:
            runtime = "NA"
        rating = tv_series.by_tag('span.value')[0].content
        artists = tv_series.by_tag('span.credit')[0].by_tag('a')
        artists = "|".join([a.content for a in artists])
        temp_res=[]
        temp_res.extend([title, genres, runtime, rating, artists])
        result.append(temp_res)
    return result
```

Now Lets Use these functions and a simple while loop to scrap all the pages
    
```py
i=1
all_data = []
while True:
    dom = get_dom(i)
    datalist=parse_dom(dom)
    if len(datalist)==0:
        break
    all_data = all_data + parse_dom(dom)
    i += 100

print "Total Elements:" + str(len(all_data))
print "First Five Elements :" + str(all_data[1:5])
```
<pre style="font-family:courier new,monospace; background-color:#f6c6529c; color:#000000">    Total Elements:898
    First Five Elements :[[u'Breaking Bad', u'Crime|Drama|Thriller', u'45 mins.', u'9.6', u'Bryan Cranston|Aaron Paul|Anna Gunn'], [u'Game of Thrones', u'Adventure|Drama|Fantasy', u'55 mins.', u'9.5', u'Lena Headey|Peter Dinklage|Maisie Williams'], [u'Planet Earth', u'Documentary', u'570 mins.', u'9.5', u'David Attenborough|Sigourney Weaver|Huw Cordey'], [u'Cosmos: A SpaceTime Odyssey', u'Documentary', u'60 mins.', u'9.5', u'Neil deGrasse Tyson|Stoney Emshwiller|Piotr Michael']]
</pre>

Voila!!! The number of elements we had to scrap were 898 and We got all of them. And to tell you, IMDB is one of the worst written HTML's. So that's Great.

In the next part of the tutorial we will run exploratory data analysis on this data using pandas and maplotlib. 

Till then keep learning.

<script src="//z-na.amazon-adsystem.com/widgets/onejs?MarketPlace=US&adInstanceId=c4ca54df-6d53-4362-92c0-13cb9977639e"></script>
