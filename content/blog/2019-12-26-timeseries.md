---
title: Using Gradient Boosting for Time Series prediction tasks
date:  2019-12-28
draft: false
url : blog/2019/12/28/timeseries/
slug: timeseries
Category: Python

Keywords:
- Pandas
- Statistics

Categories:
- Data Science

Tags:
- Machine Learning
- Data Science
- Algorithms
- Awesome Guides

description: In this post, we will try to solve the time series problem using XGBoost.

thumbnail : /images/timeseries/main.png
image :  /images/timeseries/main.png
toc : false
type : post
---

Time series prediction problems are pretty frequent in the retail domain.

Companies like Walmart and Target need to keep track of how much product should be shipped from Distribution Centres to stores. Even a small improvement in such a demand forecasting system can help save a lot of dollars in term of workforce management, inventory cost and out of stock loss.

While there are many techniques to solve this particular problem like ARIMA, Prophet, and LSTMs, we can also treat such a problem as a regression problem too and use trees to solve it.

***In this post, we will try to solve the time series problem using XGBoost.***

***The main things I am going to focus on are the sort of features such a setup takes and how to create such features.***

---

## Dataset

![](https://cdn-images-1.medium.com/max/2000/0*gLUSm0_14D8A3NvR.jpg)

Kaggle master Kazanova along with some of his friends released a [“How to win a data science competition”](https://imp.i384100.net/kjX9md) Coursera course. The Course involved a final project which itself was a time series prediction problem.

In this competition, we are given a challenging time-series dataset consisting of daily sales data, provided by one of the largest Russian software firms — 1C Company.

We have to predict total sales for every product and store in the next month.

Here is how the data looks like:

![](https://cdn-images-1.medium.com/max/2072/1*hN1eF-iQzfTp6EGg3VBAlA.png)

We are given the data at a daily level, and we want to build a model which predicts total sales for every product and store in the next month.

The variable date_block_num is a consecutive month number, used for convenience. January 2013 is 0, and October 2015 is 33. You can think of it as a proxy to month variable. I think all the other variables are self-explanatory.

***So how do we approach this sort of a problem?***


---

## Data Preparation

The main thing that I noticed is that the data preparation and [feature generation](https://towardsdatascience.com/the-hitchhikers-guide-to-feature-extraction-b4c157e96631) aspect is by far the most important thing when we attempt to solve the time series problem using regression.

### 1. Do Basic EDA and remove outliers

```py
sales = sales[sales['item_price']<100000]
sales = sales[sales['item_cnt_day']<=1000]
```

### 2. Group data at a level you want your predictions to be:

We start with creating a dataframe of distinct date_block_num, store and item combinations.

This is important because in the months we don’t have a data for an item store combination, the machine learning algorithm needs to be told explicitly that the sales are zero.

```py
from itertools import product
# Create "grid" with columns
index_cols = ['shop_id', 'item_id', 'date_block_num']

# For every month we create a grid from all shops/items combinations from that month
grid = []
for block_num in sales['date_block_num'].unique():
    cur_shops = sales.loc[sales['date_block_num'] == block_num, 'shop_id'].unique()
    cur_items = sales.loc[sales['date_block_num'] == block_num, 'item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))

grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)
grid.head()
```
![](https://cdn-images-1.medium.com/max/2000/1*yDLbk-d9EbYV7EG38MXYeg.png)

The grid dataFrame contains all the shop, items and month combinations.

We then merge the Grid with Sales to get the monthly sales DataFrame. We also replace all the NA’s with zero for months that didn’t have any sales.

```py
sales_m = sales.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': 'sum','item_price': np.mean}).reset_index()

# Merging sales numbers with the grid dataframe
sales_m = pd.merge(grid,sales_m,on=['date_block_num','shop_id','item_id'],how='left').fillna(0)

# adding the category id too from the items table.
sales_m = pd.merge(sales_m,items,on=['item_id'],how='left')
```

![](https://cdn-images-1.medium.com/max/3364/1*V_SzSZkoyGT7ce-FtPlLQw.png)

---

### 3. Create Target Encodings

To create target encodings, we group by a particular column and take the mean/min/sum etc. of the target column on it. These features are the first features we create in our model.

***Please note that these features may induce a lot of leakage/overfitting in our system and thus we don’t use them directly in our models. We will use the lag based version of these features in our models which we will create next.***

```py
groupcollist = ['item_id','shop_id','item_category_id']

aggregationlist = [('item_price',np.mean,'avg'),('item_cnt_day',np.sum,'sum'),('item_cnt_day',np.mean,'avg')]

for type_id in groupcollist:
    for column_id,aggregator,aggtype in aggregationlist:
        # get numbers from sales data and set column names
        mean_df = sales_m.groupby([type_id,'date_block_num']).aggregate(aggregator).reset_index()[[column_id,type_id,'date_block_num']]
        mean_df.columns = [type_id+'_'+aggtype+'_'+column_id,type_id,'date_block_num']
        # merge new columns on sales_m data
        sales_m = pd.merge(sales_m,mean_df,on=['date_block_num',type_id],how='left')
```
We group by item_id, shop_id, and item_category_id and aggregate on the item_price and item_cnt_day column to create the following new features:

![We create the highlighted target encodings](https://cdn-images-1.medium.com/max/2804/1*TNJdVv0Bka75S5QKHV0D-w.png)

We could also have used [featuretools](https://towardsdatascience.com/the-hitchhikers-guide-to-feature-extraction-b4c157e96631) for this. **Featuretools** is a framework to perform automated feature engineering. It excels at transforming temporal and relational datasets into feature matrices for machine learning.

---

### 4. Create Lag Features

The next set of features our model needs are the lag based Features.

When we create regular classification models, we treat training examples as fairly independent of each other. But in case of time series problems, at any point in time, the model needs information on what happened in the past.

We can’t do this for all the past days, but we can provide the models with the most recent information nonetheless using our target encoded features.

```py

lag_variables  = ['item_id_avg_item_price','item_id_sum_item_cnt_day','item_id_avg_item_cnt_day','shop_id_avg_item_price','shop_id_sum_item_cnt_day','shop_id_avg_item_cnt_day','item_category_id_avg_item_price','item_category_id_sum_item_cnt_day','item_category_id_avg_item_cnt_day','item_cnt_day']
lags = [1 ,2 ,3 ,4, 5, 12]
# we will keep the results in thsi dataframe
sales_means = sales_m.copy()
for lag in lags:
    sales_new_df = sales_m.copy()
    sales_new_df.date_block_num+=lag
    # subset only the lag variables we want
    sales_new_df = sales_new_df[['date_block_num','shop_id','item_id']+lag_variables]
    sales_new_df.columns = ['date_block_num','shop_id','item_id']+ [lag_feat+'_lag_'+str(lag) for lag_feat in lag_variables]
    # join with date_block_num,shop_id and item_id
    sales_means = pd.merge(sales_means, sales_new_df,on=['date_block_num','shop_id','item_id'] ,how='left')
```
So we aim to add past information for a few features in our data. We do it for all the new features we created and the item_cnt_day feature.

We fill the NA’s with zeros once we have the lag features.

```py
for feat in sales_means.columns:
    if 'item_cnt' in feat:
        sales_means[feat]=sales_means[feat].fillna(0)
    elif 'item_price' in feat:
    sales_means[feat]=sales_means[feat].fillna(sales_means[feat].median())
```

We end up creating a lot of lag features with different lags:

    'item_id_avg_item_price_lag_1','item_id_sum_item_cnt_day_lag_1', 'item_id_avg_item_cnt_day_lag_1','shop_id_avg_item_price_lag_1', 'shop_id_sum_item_cnt_day_lag_1','shop_id_avg_item_cnt_day_lag_1','item_category_id_avg_item_price_lag_1','item_category_id_sum_item_cnt_day_lag_1','item_category_id_avg_item_cnt_day_lag_1', 'item_cnt_day_lag_1',

    'item_id_avg_item_price_lag_2', 'item_id_sum_item_cnt_day_lag_2','item_id_avg_item_cnt_day_lag_2', 'shop_id_avg_item_price_lag_2','shop_id_sum_item_cnt_day_lag_2', 'shop_id_avg_item_cnt_day_lag_2','item_category_id_avg_item_price_lag_2','item_category_id_sum_item_cnt_day_lag_2','item_category_id_avg_item_cnt_day_lag_2', 'item_cnt_day_lag_2',

    ...

---

## Modelling

### 1. Drop the unrequired columns

As previously said, we are going to drop the target encoded features as they might induce a lot of overfitting in the model. We also lose the item_name and item_price feature.

```py

cols_to_drop = lag_variables[:-1] + ['item_name','item_price']

for col in cols_to_drop:
    del sales_means[col]
```
### 2. Take a recent bit of data only

When we created the lag variables, we induced a lot of zeroes in the system. We used the maximum lag as 12. To counter that we remove the first 12 months indexes.

```py
sales_means = sales_means[sales_means['date_block_num']>11]
```

### 3. Train and CV Split

When we do a time series split, we usually don’t take a cross-sectional split as the data is time-dependent. We want to create a model that sees till now and can predict the next month well.

```py
X_train = sales_means[sales_means['date_block_num']<33]
X_cv =  sales_means[sales_means['date_block_num']==33]

Y_train = X_train['item_cnt_day']
Y_cv = X_cv['item_cnt_day']

del X_train['item_cnt_day']
del X_cv['item_cnt_day']
```

### 4. Create Baseline

![](https://cdn-images-1.medium.com/max/2000/0*Ujr-irYHlXd0SAGP.jpg)

Before we proceed with modelling steps, lets check the RMSE of a naive model, as we want to [have an RMSE to compare](https://towardsdatascience.com/take-your-machine-learning-models-to-production-with-these-5-simple-steps-35aa55e3a43c) to. We assume that we are going to predict the last month sales as current month sale for our baseline model. We can quantify the performance of our model using this baseline RMSE.

```py
from sklearn.metrics import mean_squared_error
sales_m_test = sales_m[sales_m['date_block_num']==33]

preds = sales_m.copy()
preds['date_block_num']=preds['date_block_num']+1
preds = preds[preds['date_block_num']==33]
preds = preds.rename(columns={'item_cnt_day':'preds_item_cnt_day'})
preds = pd.merge(sales_m_test,preds,on = ['shop_id','item_id'],how='left')[['shop_id','item_id','preds_item_cnt_day','item_cnt_day']].fillna(0)

# We want our predictions clipped at (0,20). Competition Specific
preds['item_cnt_day'] = preds['item_cnt_day'].clip(0,20)
preds['preds_item_cnt_day'] = preds['preds_item_cnt_day'].clip(0,20)
baseline_rmse = np.sqrt(mean_squared_error(preds['item_cnt_day'],preds['preds_item_cnt_day']))

print(baseline_rmse)
```

    1.1358170090812756

---

### 5. Train XGB

We use the XGBRegressor object from the xgboost scikit API to build our model. Parameters are taken from this [kaggle kernel](https://www.kaggle.com/dlarionov/feature-engineering-xgboost). If you have time, you can use hyperopt to [automatically find out the hyperparameters](https://towardsdatascience.com/automate-hyperparameter-tuning-for-your-models-71b18f819604) yourself.

```py
from xgboost import XGBRegressor

model = XGBRegressor(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300,
    colsample_bytree=0.8,
    subsample=0.8,
    eta=0.3,    
    seed=42)

model.fit(
    X_train,
    Y_train,
    eval_metric="rmse",
    eval_set=[(X_train, Y_train), (X_cv, Y_cv)],
    verbose=True,
    early_stopping_rounds = 10)
```

![](https://cdn-images-1.medium.com/max/2288/1*uK9IjFKLEWPGbbeDLc2hfg.png)

After running this, we can see RMSE in ranges of ***0.93*** on the CV set. And that is pretty impressive based on our baseline validation RMSE of ***1.13***. And so we work on deploying this model as part of our [continuous integration](https://towardsdatascience.com/take-your-machine-learning-models-to-production-with-these-5-simple-steps-35aa55e3a43c) effort.

### 5. Plot Feature Importance

We can also see the important features that come from XGB.

```py
feature_importances = pd.DataFrame({'col': columns,'imp':model.feature_importances_})
feature_importances = feature_importances.sort_values(by='imp',ascending=False)
px.bar(feature_importances,x='col',y='imp')
```
![Feature importances](https://cdn-images-1.medium.com/max/3716/1*TZ_BawTl6O1kMuTUTMYoHw.png)

---

## Conclusion

In this post, we talked about how we can use trees for even time series modelling. The purpose was not to get perfect scores on the kaggle leaderboard but to gain an understanding of how such models work.

![](https://cdn-images-1.medium.com/max/3004/0*vsyzeBzrG4q4Z33z.png)

When I took part in this competition as part of the [course](https://imp.i384100.net/kjX9md), a couple of years back, using trees I reached near the top of the leaderboard.

Over time people have worked a lot on tweaking the model, hyperparameter tuning and creating even more informative features. But the basic approach has remained the same.

You can find the whole running code on [GitHub](https://github.com/MLWhiz/data_science_blogs/tree/master/time_series_xgb).

Take a look at the [How to Win a Data Science Competition: Learn from Top Kagglers](https://imp.i384100.net/kjX9md) course in the [Advanced machine learning specialization](https://imp.i384100.net/kjX9md) by Kazanova. This course talks about a lot of ways to improve your models using feature engineering and hyperparameter tuning.

I am going to be writing more beginner-friendly posts in the future too. Let me know what you think about the series. Follow me up at [**Medium**](https://mlwhiz.medium.com/) or Subscribe to my [**blog**](mlwhiz.com).

Also, a small disclaimer — There might be some affiliate links in this post to relevant resources, as sharing knowledge is never a bad idea.
