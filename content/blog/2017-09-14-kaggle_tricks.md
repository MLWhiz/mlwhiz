---
title: "Good Feature Building Techniques - Tricks for Kaggle -  My Kaggle Code Repository"
date:  2017-09-14
draft: false
url : blog/2017/09/14/kaggle_tricks/
slug: kaggle_tricks
Category: Python, NLP, Algorithms, Kaggle
Keywords:
- Python
- NLP
- Algorithms
- Kaggle
Tags:
- Machine Learning
- Data Science
- Kaggle
Categories:
- Data Science


description:  Here is the list of ideas I gather in day to day life, where people have used creativity to get great results on Kaggle
toc : false
thumbnail: /images/category_bgs/default_bg.jpg
image: /images/category_bgs/default_bg.jpg
---

Often times it happens that we fall short of creativity. And creativity is one of the basic ingredients of what we do. Creating features needs creativity. So here is the list of ideas I gather in day to day life, where people have used creativity to get great results on Kaggle leaderboards.

Take a look at the [How to Win a Data Science Competition: Learn from Top Kagglers](https://imp.i384100.net/kjX9md) course in the [Advanced machine learning specialization](https://imp.i384100.net/kjX9md) by Kazanova(Number 3 Kaggler at the time of writing). You can start for free with the 7-day Free Trial.


This post is inspired by a [Kernel](https://www.kaggle.com/gaborfodor/from-eda-to-the-top-lb-0-368) on Kaggle written by Beluga, one of the top Kagglers, for a knowledge based [competition](https://www.kaggle.com/c/nyc-taxi-trip-duration).

Some of the techniques/tricks I am sharing have been taken directly from that kernel so you could take a look yourself.
Otherwise stay here and read on.

## 1. Don't try predicting the future when you don't have to:

If both training/test comes from the same timeline, we can get really crafty with features. Although this is a case with Kaggle only, we can use this to our advantage. For example: In the Taxi Trip duration challenge the test data is randomly sampled from the train data. In this case we can use the target variable averaged over different categorical variable as a feature. Like in this case Beluga actually used the averaged the target variable over different weekdays. He then mapped the same averaged value as a variable by mapping it to test data too.

## 2. logloss clipping Technique:

Something that I learned in the Neural Network course by Jeremy Howard. Its based on a very simple Idea. Logloss penalises a lot if we are very confident and wrong. So in case of Classification problems where we have to predict probabilities, it would be much better to clip our probabilities between 0.05-0.95 so that we are never very sure about our prediction.

## 3. kaggle submission in gzip format:

A small piece of code that will help you save countless hours of uploading. Enjoy.
df.to_csv('submission.csv.gz', index=False, compression='gzip')

## 4. How best to use Latitude and Longitude features - Part 1:

One of the best things that I liked about the Beluga Kernel is how he used the Lat/Lon Data. So in the example we had pickup Lat/Lon and Dropoff Lat/Lon. We created features like:

#### A. Haversine Distance Between the Two Lat/Lons:

```py
def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h
```

#### B. Manhattan Distance Between the two Lat/Lons:

```py
def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b
```

#### C. Bearing Between the two Lat/Lons:

```py
def bearing_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))
```

#### D. Center Latitude and Longitude between Pickup and Dropoff:

```py
train.loc[:, 'center_latitude'] = (train['pickup_latitude'].values + train['dropoff_latitude'].values) / 2
train.loc[:, 'center_longitude'] = (train['pickup_longitude'].values + train['dropoff_longitude'].values) / 2
```

## 5. How best to use Latitude and Longitude features - Part 2:
The Second way he used the Lat/Lon Feats was to create clusters for Pickup and Dropoff Lat/Lons. The way it worked was it created sort of Boroughs in the data by design.

```py
from sklearn.cluster import MiniBatchKMeans
coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                    train[['dropoff_latitude', 'dropoff_longitude']].values,
                    test[['pickup_latitude', 'pickup_longitude']].values,
                    test[['dropoff_latitude', 'dropoff_longitude']].values))

sample_ind = np.random.permutation(len(coords))[:500000]
kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])

train.loc[:, 'pickup_cluster'] = kmeans.predict(train[['pickup_latitude', 'pickup_longitude']])
train.loc[:, 'dropoff_cluster'] = kmeans.predict(train[['dropoff_latitude', 'dropoff_longitude']])
test.loc[:, 'pickup_cluster'] = kmeans.predict(test[['pickup_latitude', 'pickup_longitude']])
test.loc[:, 'dropoff_cluster'] = kmeans.predict(test[['dropoff_latitude', 'dropoff_longitude']])
```

He then used these Clusters to create features like counting no of trips going out and coming in on a particular day.

## 6. How best to use Latitude and Longitude features - Part 3

He used PCA to transform longitude and latitude coordinates. In this case it is not about dimension reduction since he transformed 2D-> 2D. The rotation could help for decision tree splits, and it did actually.

```py
pca = PCA().fit(coords)
train['pickup_pca0'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 0]
train['pickup_pca1'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 1]
train['dropoff_pca0'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
train['dropoff_pca1'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
test['pickup_pca0'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 0]
test['pickup_pca1'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 1]
test['dropoff_pca0'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
test['dropoff_pca1'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
```

## 7. Lets not forget the Normal Things you can do with your features:

- Scaling by Max-Min
- Normalization using Standard Deviation
- Log based feature/Target: use log based features or log based target function.
- One Hot Encoding

## 8. Creating Intuitive Additional Features:

A) Date time Features: Time based Features like "Evening", "Noon", "Night", "Purchases_last_month", "Purchases_last_week" etc.

B) Thought Features: Suppose you have shopping cart data and you want to categorize TripType (See Walmart Recruiting: Trip Type Classification on [Kaggle](https://www.kaggle.com/c/walmart-recruiting-trip-type-classification/) for some background).

You could think of creating a feature like "Stylish" where you create this variable by adding together number of items that belong to category Men's Fashion, Women's Fashion, Teens Fashion.

You could create a feature like "Rare" which is created by tagging some items as rare, based on the data we have and then counting the number of those rare items in the shopping cart. Such features might work or might not work. From what I have observed they normally provide a lot of value.

I feel this is the way that Target's "Pregnant Teen model" was made. They would have had a variable in which they kept all the items that a pregnant teen could buy and put it into a classification algorithm.

## 9 . The not so Normal Things which people do:

These features are highly unintuitive and should not be created where the machine learning model needs to be interpretable.

A) Interaction Features: If you have features A and B create features A*B, A+B, A/B, A-B. This explodes the feature space. If you have 10 features and you are creating two variable interactions you will be adding 10C2 * 4  features = 180 features to your model. And most of us have a lot more than 10 features.

B) Bucket Feature Using Hashing: Suppose you have a lot of features. In the order of Thousands but you don't want to use all the thousand features because of the training times of algorithms involved. People bucket their features using some hashing algorithm to achieve this.Mostly done for text classification tasks.
For example:
If we have 6 features A,B,C,D,E,F.
And the row of data is:
A:1,B:1,C:1,D:0,E:1,F:0
I may decide to use a hashing function so that these 6 features correspond to 3 buckets and create the data using this feature hashing vector.
After processing my data might look like:
Bucket1:2,Bucket2:2,Bucket3:0
Which happened because A and B fell in bucket1, C and E fell in bucket2 and D and F fell in bucket 3. I summed up the observations here, but you could substitute addition with any math function you like.
Now i would use Bucket1,Bucket2,Bucket3 as my variables for machine learning.


Will try to keep on expanding. Wait for more....
