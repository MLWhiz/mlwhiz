---
title: "Minimal Pandas Subset for Data Scientists"
date:  2019-07-20
draft: false
url : blog/2019/07/20/pandas_subset/
slug: pandas_subset
Category: Python

Keywords:
- Pandas

Categories:
- Awesome Guides
- Data Science

Tags:
- Python
- Awesome Guides
- Best Content
- Machine Learning
- Data Science
- Productivity

description: There are multiple ways to doing the same thing in Pandas, and that might make it troublesome for the beginner user.This post is about handling most of the data manipulation cases in Python using a straightforward, simple, and matter of fact way.

thumbnail : /images/pandas_subset/1.jpeg
image : /images/pandas_subset/1.jpeg
toc : false
type : post
---


Pandas is a vast library.

Data manipulation is a breeze with pandas, and it has become such a standard for it that a lot of parallelization libraries like Rapids and Dask are being created in line with Pandas syntax.

Still, I generally have some issues with it.

***There are multiple ways to doing the same thing in Pandas, and that might make it troublesome for the beginner user.***

This has inspired me to come up with a minimal subset of pandas functions I use while coding.

I have tried it all, and currently, I stick to a particular way. It is like a mind map.

***Sometimes because it is fast and sometimes because it’s more readable and sometimes because I can do it with my current knowledge. And sometimes because I know that a particular way will be a headache in the long run(think multi-index)***

***This post is about handling most of the data manipulation cases in Python using a straightforward, simple, and matter of fact way.***

With a sprinkling of some recommendations throughout.

I will be using a data set of 1,000 popular movies on IMDB in the last ten years. You can also follow along in the [Kaggle Kernel](https://www.kaggle.com/mlwhiz/minimal-pandas-subset).

---

## Some Default Pandas Requirements

![](/images/pandas_subset/2.png)

As good as the Jupyter notebooks are, some things still need to be specified when working with Pandas.

***Sometimes your notebook won’t show you all the columns. Sometimes it will display all the rows if you print the dataframe. ***You can control this behavior by setting some defaults of your own while importing Pandas. You can automate it using [this addition](https://towardsdatascience.com/three-great-additions-for-your-jupyter-notebooks-cd7373b00e96) to your notebook.

For instance, this is the setting I use.

```py
import pandas as pd
# pandas defaults
pd.options.display.max_columns = 500
pd.options.display.max_rows = 500
```

---

## Reading Data with Pandas

![](/images/pandas_subset/3.jpg)


The first thing we do is reading the data source and so here is the code for that.

```py
df = pd.read_csv("IMDB-Movie-Data.csv")
```

***Recommendation:*** I could also have used pd.read_table to read the file. The thing is that pd.read_csv has default separator as , and thus it saves me some code. I also genuinely don’t understand the use of pd.read_table

If your data is in some SQL Datasource, you could have used the following code. You get the results in the dataframe format.

```py
# Reading from SQL Datasource
import MySQLdb
from pandas import DataFrame
from pandas.io.sql import read_sql

db = MySQLdb.connect(host="localhost",    # your host, usually localhost
                     user="root",         # your username
                     passwd="password",   # your password
                     db="dbname")         # name of the data base

query = "SELECT * FROM tablename"

df = read_sql(query, db)
```
---

## Data Snapshot

![](/images/pandas_subset/4.jpg)


Always useful to see some of the data.

You can use simple head and tail commands with an option to specify the number of rows.

```py
# top 5 rows
df.head()

# top 50 rows
df.head(50)

# last 5 rows
df.tail()

# last 50 rows
df.tail(50)
```
You can also see simple dataframe statistics with the following commands.

```py
# To get statistics of numerical columns
df.describe()
```

![](/images/pandas_subset/4a.png)

```py
# To get maximum value of a column. When you take a single column you can think of it as a list and apply functions you would apply to a list. You can also use min for instance.

print(max(df['rating']))

# no of rows in dataframe
print(len(df))

# Shape of Dataframe
print(df.shape)
```

    9.0
    1000
    (1000,12)

***Recommendation:*** Generally working with Jupyter notebook,***I make it a point of having the first few cells in my notebook containing these snapshots*** of the data. This helps me see the structure of the data whenever I want to. If I don’t follow this practice, I notice that I end up repeating the `.head()` command a lot of times in my code.

---

## Handling Columns in Dataframes

![](/images/pandas_subset/5.jpg)

### a. Selecting a column

For some reason Pandas lets you choose columns in two ways. Using the dot operator like `df.Title` and using square brackets like `df['Title']`

I prefer the second version, mostly. Why?

There are a couple of reasons you would be better off with the square bracket version in the longer run.

* If your column name contains spaces, then the dot version won’t work. For example, `df.Revenue (Millions)` won’t work while `df['Revenue (Millions)']` will.

* It also won’t work if your column name is `count` or `mean` or any of pandas predefined functions.

* Sometimes you might need to create a for loop over your column names in which your column name might be in a variable. In that case, the dot notation will not work. For Example, This works:

```py
colname = 'height'
df[colname]
```

While this doesn’t:

```py
colname = 'height'
df.colname
```

Trust me. Saving a few characters is not worth it.

***Recommendation: Stop using the dot operator***. It is a construct that originated from a different language(R) and respectfully should be left there.

### b. Getting Column Names in a list

You might need a list of columns for some later processing.

```py
columnnames = df.columns
```

### c. Specifying user-defined Column Names:

Sometimes you want to change the column names as per your taste. I don’t like spaces in my column names, so I change them as such.

```py
df.columns = ['Rank', 'Title', 'Genre', 'Description', 'Director', 'Actors', 'Year',
       'Runtime_Minutes', 'Rating', 'Votes', 'Revenue_Millions',
       'Metascore']
```

I could have used another way.

This is the one case where both of the versions are important. When I have to change a lot of column names, I go with the way above. When I have to change the name of just one or two columns I use:

```py
df.rename(columns = {'Revenue (Millions)':'Rev_M','Runtime (Minutes)':'Runtime_min'},inplace=True)
```

### d. Subsetting specific columns:

Sometimes you only need to work with particular columns in a dataframe. e.g., to separate numerical and categorical columns, or remove unnecessary columns. Let’s say in our example; we don’t need the description, director, and actor column.

```py
df = df[['Rank', 'Title', 'Genre', 'Year','Runtime_min', 'Rating', 'Votes', 'Rev_M', 'Metascore']]
```

### e. Seeing column types:

Very useful while debugging. If your code throws an error that you cannot add a str and int, you will like to run this command.

```py
df.dtypes
```

---

## Applying Functions on DataFrame: Apply and Lambda

![](/images/pandas_subset/6.png)

`apply` and `lambda` are some of the best things I have learned to use with pandas.

I use `apply` and `lambda` anytime I get stuck while building a complex logic for a new column or filter.

### a. Creating a Column

You can create a new column in many ways.

If you want a column that is a sum or difference of columns, you can pretty much use simple basic arithmetic. Here I get the average rating based on IMDB and Normalized Metascore.

```py
df['AvgRating'] = (df['Rating'] + df['Metascore']/10)/2
```

But sometimes we may need to build complex logic around the creation of new columns.

To give you a convoluted example, let’s say that we want to build a custom movie score based on a variety of factors.

***Say, If the movie is of the thriller genre, I want to add 1 to the IMDB rating subject to the condition that IMDB rating remains less than or equal to 10. And If a movie is a comedy I want to subtract one from the rating.***

***How do we do that?***

Whenever I get a hold of such complex problems, I use `apply/lambda`. Let me first show you how I will do this.

```py
def custom_rating(genre,rating):
    if 'Thriller' in genre:
        return min(10,rating+1)
    elif 'Comedy' in genre:
        return max(0,rating-1)
    else:
        return rating

df['CustomRating'] = df.apply(lambda x: custom_rating(x['Genre'],x['Rating']),axis=1)
```

The general structure is:

* You define a function that will take the column values you want to play with to come up with your logic. Here the only two columns we end up using are genre and rating.

* You use an apply function with lambda along the row with axis=1. The general syntax is:

```py
df.apply(lambda x: func(x['col1'],x['col2']),axis=1)
```

You should be able to create pretty much any logic using apply/lambda since you just have to worry about the custom function.

### b. Filtering a dataframe

![](/images/pandas_subset/7.jpeg)

Pandas make filtering and subsetting dataframes pretty easy. You can filter and subset dataframes using normal operators and `&,|,~` operators.

```py
# Single condition: dataframe with all movies rated greater than 8

df_gt_8 = df[df['Rating']>8]

# Multiple conditions: AND - dataframe with all movies rated greater than 8 and having more than 100000 votes

And_df = df[(df['Rating']>8) & (df['Votes']>100000)]

# Multiple conditions: OR - dataframe with all movies rated greater than 8 or having a metascore more than 90

Or_df = df[(df['Rating']>8) | (df['Metascore']>80)]

# Multiple conditions: NOT - dataframe with all emovies rated greater than 8 or having a metascore more than 90 have to be excluded

Not_df = df[~((df['Rating']>8) | (df['Metascore']>80))]
```
Pretty simple stuff.

But sometimes we may need to do complex filtering operations.

And sometimes we need to do some operations which we won’t be able to do using just the above format.

For instance: Let us say ***we want to filter those rows where the number of words in the movie title is greater than or equal to than 4.***

***How would you do it?***

Trying the below will give you an error. Apparently, you cannot do anything as simple as split with a series.

```py
new_df = df[len(df['Title'].split(" "))>=4]
```

    AttributeError: 'Series' object has no attribute 'split'

One way is first to create a column which contains no of words in the title using apply and then filter on that column.

```py
#create a new column
df['num_words_title'] = df.apply(lambda x : len(x['Title'].split(" ")),axis=1)

#simple filter on new column
new_df = df[df['num_words_title']>=4]
```

And that is a perfectly fine way as long as you don’t have to create a lot of columns. But I prefer this:

```py
new_df = df[df.apply(lambda x : len(x['Title'].split(" "))>=4,axis=1)]
```

What I did here is that ***my apply function returns a boolean which can be used to filter.***

Now once you understand that you just have to create a column of booleans to filter, you can use any function/logic in your apply statement to get however complex a logic you want to build.

Let us see another example. I will try to do something a little complex to show the structure.

***We want to find movies for which the revenue is less than the average revenue for that particular year?***

```py
year_revenue_dict = df.groupby(['Year']).agg({'Rev_M':np.mean}).to_dict()['Rev_M']

def bool_provider(revenue, year):
    return revenue<year_revenue_dict[year]

new_df = df[df.apply(lambda x : bool_provider(x['Rev_M'],x['Year']),axis=1)]
```

We have a function here which we can use to write any logic. That provides a lot of power for advanced filtering as long as we can play with simple variables.

### c. Change Column Types

I even use apply to change the column types since I don’t want to remember the syntax for changing column type and also since it lets me do much more complicated things.

The usual syntax to change column type is astype in Pandas. So if I had a column named price in my data in an str format. I could do this:

```py
df['Price'] = newDf['Price'].astype('int')
```

But sometimes it won’t work as expected.

You might get the error: `ValueError: invalid literal for long() with base 10: ‘13,000’`. That is you cannot cast a string with “,” to an int. To do that we first have to get rid of the comma.

After facing this problem time and again, I have stopped using astype altogether now and just use apply to change column types.

```py
df['Price'] = df.apply(lambda x: int(x['Price'].replace(',', '')),axis=1)
```

### And lastly, there is progress_apply

![](/images/pandas_subset/8.png)

`progress_apply` is a function that comes with `tqdm` package.

And this has saved me a lot of time.

Sometimes when you have got a lot of rows in your data, or you end up writing a pretty complex apply function, you will see that apply might take a lot of time.

I have seen apply taking hours when working with Spacy. In such cases, you might like to see the progress bar with `apply`.

You can use `tqdm` for that.

After the initial imports at the top of your notebook, just replace `apply` with `progress_apply` and everything remains the same.

```py
from tqdm import tqdm, tqdm_notebook
tqdm_notebook().pandas()

df.progress_apply(lambda x: custom_rating_function(x['Genre'],x['Rating']),axis=1)
```

And you get progress bars.

![](/images/pandas_subset/9.png)

***Recommendation:***vWhenever you see that you have to create a column with custom complex logic, think of apply and lambda. Try using progress_apply too.

---

## Aggregation on Dataframes: groupby

![](/images/pandas_subset/10.jpg)

groupby will come up a lot of times whenever you want to aggregate your data. Pandas lets you do this efficiently with the groupby function.

There are a lot of ways that you can use groupby. I have seen a lot of versions, but I prefer a particular style since I feel the version I use is easy, intuitive, and scalable for different use cases.

```py
df.groupby(list of columns to groupby on).aggregate({'colname':func1, 'colname2':func2}).reset_index()
```

Now you see it is pretty simple. You just have to worry about supplying two primary pieces of information.

* List of columns to groupby on, and

* A dictionary of columns and functions you want to apply to those columns

reset_index() is a function that resets the index of a dataframe. I apply this function ALWAYS whenever I do a groupby, and you might think of it as a default syntax for groupby operations.

Let us check out an example.

```py
# Find out the sum of votes and revenue by year

import numpy as np
df.groupby(['Year']).aggregate({'Votes':np.sum, 'Rev_M':np.sum}).reset_index()
```

![](/images/pandas_subset/11.png)

You might also want to group by more than one column. It is fairly straightforward.

```py
df.groupby(['Year','Genre']).aggregate({'Votes':np.sum, 'Rev_M':np.sum}).reset_index()
```

![](/images/pandas_subset/12.png)

***Recommendation:*** Stick to one syntax for groupby. Pick your own if you don’t like mine but stick to one.

---

## Dealing with Multiple Dataframes: Concat and Merge:

![](/images/pandas_subset/13.jpeg)

### a. concat

Sometimes we get data from different sources. Or someone comes to you with multiple files with each file having data for a particular year.

***How do we create a single dataframe from a single dataframe?***

Here we will create our use case artificially since we just have a single file. We are creating two dataframes first using the basic filter operations we already know.

```py
movies_2006 = df[df['Year']==2006]
movies_2007 = df[df['Year']==2007]
```

Here we start with two dataframes: `movies_2006` containing info for movies released in 2006 and `movies_2007` containing info for movies released in 2007. We want to create a single dataframe that includes movies from both 2006 and 2007

```py
movies_06_07 = pd.concat([movies_2006,movies_2007])
```

### b. merge

Most of the data that you will encounter will never come in a single file. One of the files might contain ratings for a particular movie, and another might provide the number of votes for a movie.

In such a case we have two dataframes which need to be merged so that we can have all the information in a single view.

Here we will create our use case artificially since we just have a single file. We are creating two dataframes first using the basic column subset operations we already know.

```py
rating_dataframe = df[['Title','Rating']]
votes_dataframe =  df[['Title','Votes']]
```

![](/images/pandas_subset/14.png)

We need to have all this information in a single dataframe. How do we do this?

```py
rating_vote_df = pd.merge(rating_dataframe,votes_dataframe,on='Title',how='left')

rating_vote_df.head()
```

![](/images/pandas_subset/15.png)

We provide this merge function with four attributes- 1st DF, 2nd DF, join on which column and the joining criteria:`['left','right','inner','outer']`

***Recommendation:*** I usually always end up using left join. You will rarely need to join using outer or right. Actually whenever you need to do a right join you actually just really need a left join with the order of dataframes reversed in the merge function.

---

## Reshaping Dataframes: Melt and pivot_table(reverseMelt)

![](/images/pandas_subset/16.jpeg)

Most of the time, we don’t get data in the exact form we want.

For example, sometimes we might have data in columns which we might need in rows.

Let us create an artificial example again. You can look at the code below that I use to create the example, but really it doesn’t matter.

```py
genre_set = set()
for genre in df['Genre'].unique():
    for g in genre.split(","):
        genre_set.add(g)
for genre in genre_set:
    df[genre] = df['Genre'].apply(lambda x: 1 if genre in x else 0)

working_df = df[['Title','Rating', 'Votes',
       'Rev_M']+list(genre_set)]

working_df.head()
```

So we start from a `working_df` like this:

![](/images/pandas_subset/17.png)

Now, this is not particularly a great structure to have data in. We might like it better if we had a dataframe with only one column Genre and we can have multiple rows repeated for the same movie. So the movie ‘Prometheus’ might be having three rows since it has three genres. How do we make that work?

We use `melt`:

```py
reshaped_df = pd.melt(working_df,id_vars = ['Title','Rating','Votes','Rev_M'],value_vars = list(genre_set),var_name = 'Genre', value_name ='Flag')

reshaped_df.head()
```

![](/images/pandas_subset/18.png)

So in this melt function, we provided five attributes:

* dataframe_name = working_df

* id_vars: List of vars we want in the current form only.

* value_vars: List of vars we want to melt/put in the same column

* var_name: name of the column for value_vars

* value_name: name of the column for value of value_vars

There is still one thing remaining. For Prometheus, we see that it is a thriller and the flag is 0. The flag 0 is unnecessary data we can filter out, and we will have our results. We keep only the genres with flag 1

```py
reshaped_df  = reshaped_df[reshaped_df['Flag']==1]
```
![](/images/pandas_subset/19.png)

***What if we want to go back?***

We need the values in a column to become multiple columns. How? We use `pivot_table`

```py
re_reshaped_df = reshaped_df.pivot_table(index=['Title','Rating','Votes','Rev_M'], columns='Genre',
                        values='Flag', aggfunc='sum').reset_index()

re_reshaped_df.head()
```

![](/images/pandas_subset/20.png)

We provided four attributes to the pivot_table function.

* index: We don’t want to change these column structures

* columns: explode this column into multiple columns

* values: use this column to aggregate

* aggfunc: the aggregation function.

We can then fill the missing values by 0 using `fillna`

```py
re_reshaped_df=re_reshaped_df.fillna(0)
```

![](/images/pandas_subset/21.png)

***Recommendation:*** Multiple columns to one column: melt and One column to multiple columns: pivot_table . There are other ways to do melt — stack and different ways to do pivot_table: pivot,unstack. Stay away from them and just use melt and pivot_table. There are some valid reasons for this like unstack and stack will create multi-index and we don’t want to deal with that, and pivot cannot take multiple columns as the index.

## Conclusion

![](/images/pandas_subset/22.jpg)

> With Pandas, less choice is more

***Here I have tried to profile some of the most useful functions in pandas I end up using most often.***

Pandas is a vast library with a lot of functionality and custom options. That makes it essential that you should have a mindmap where you stick to a particular syntax for a specific thing.

***Here I have shared mine, and you can proceed with it and make it better as your understanding of the library grows.***

I hope you found this post useful and worth your time. I tried to make this as simple as possible, but you may always **ask me** or see the documentation for doubts.

Whole code and data are posted in the [Kaggle Kernel](https://www.kaggle.com/mlwhiz/minimal-pandas-subset).

Also, if you want to learn more about Python 3, I would like to call out an excellent course on Learn [Intermediate level Python](https://imp.i384100.net/6yyWGV) from the University of Michigan. Do check it out.

I am going to be writing more of such posts in the future too. Let me know what you think about them. Follow me up at [**Medium**](https://mlwhiz.medium.com/) or Subscribe to my [**blog**](mlwhiz.com).
