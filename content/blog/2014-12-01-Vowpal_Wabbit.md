---
title: "Exploring Vowpal Wabbit with the Avazu Clickthrough Prediction Challenge"
date:  2014-12-01
draft: false
url : blog/2014/12/01/exploring_vowpal_wabbit_avazu/
aliases:
- blog/2014/12/01/Exploring_Vowpal_Wabbit_with_Avazu/
Category: Python
Keywords:
- VW
- Vowpal Wabbit
- Avazu Clickthrough
Tags:
- Machine Learning
- Data Science
- Python
description: Exploring Vowpal Wabbit with the Avazu Clickthrough Prediction Challenge
toc : false

Categories:
- Data Science

type : post
thumbnail: /images/category_bgs/default_bg.jpg
image: /images/category_bgs/default_bg.jpg

---

In online advertising, click-through rate (CTR) is a very important metric for evaluating ad performance. As a result, click prediction systems are essential and widely used for sponsored search and real-time bidding.

For this competition, we have provided 11 days worth of Avazu data to build and test prediction models. Can you find a strategy that beats standard classification algorithms? The winning models from this competition will be released under an open-source
license.

## Data Fields

<pre style="font-family:courier new,monospace; background-color:#f6c6529c; color:#000000">
id: ad identifier
click: 0/1 for non-click/click
hour: format is YYMMDDHH, so 14091123 means 23:00 on Sept. 11, 2014 UTC.
C1 -- anonymized categorical variable
banner_pos
site_id
site_domain
site_category
app_id
app_domain
app_category
device_id
device_ip
device_model
device_type
device_conn_type
C14-C21 -- anonymized categorical variables
</pre>

## Loading Data

```py
## Loading the data

import pandas as pd
import numpy as np
import string as stri

#too large data not keeping it in memory.
# will be using line by line scripting.
#data = pd.read_csv("/Users/RahulAgarwal/kaggle_cpr/train")
```

Since the data is too large around 6 gb , we will proceed by doing line by line analysis of data. We will try to use vowpal wabbit first of all as it is an online model and it also gives us the option of minimizing log loss as a default. It is also very fast to run and will give us quite an intuition as to how good our prediction can be.

I will use all the variables in the first implementation and we will rediscover things as we move on

## Running Vowpal Wabbit
## Creating data in vowpal format (One Time Only)

```py
from datetime import datetime

def csv_to_vw(loc_csv, loc_output, train=True):
    start = datetime.now()
    print("\nTurning %s into %s. Is_train_set? %s"%(loc_csv,loc_output,train))
    i = open(loc_csv, "r")
    j = open(loc_output, 'wb')
    counter=0
    with i as infile:
        line_count=0
        for line in infile:
            # to counter the header
            if line_count==0:
                line_count=1
                continue
            # The data has all categorical features
            #numerical_features = ""
            categorical_features = ""
            counter = counter+1
            #print counter
            line = line.split(",")
            if train:
                #working on the date column. We will take day , hour
                a = line[2]
                new_date= datetime(int("20"+a[0:2]),int(a[2:4]),int(a[4:6]))
                day = new_date.strftime("%A")
                hour= a[6:8]
                categorical_features += " |hr %s" % hour
                categorical_features += " |day %s" % day
                # 24 columns in data    
                for i in range(3,24):
                    if line[i] != "":
                        categorical_features += "|c%s %s" % (str(i),line[i])
            else:
                a = line[1]
                new_date= datetime(int("20"+a[0:2]),int(a[2:4]),int(a[4:6]))
                day = new_date.strftime("%A")
                hour= a[6:8]
                categorical_features += " |hr %s" % hour
                categorical_features += " |day %s" % day
                for i in range(2,23):
                    if line[i] != "":
                        categorical_features += " |c%s %s" % (str(i+1),line[i])
  #Creating the labels
            #print "a"
            if train: #we care about labels
                if line[1] == "1":
                    label = 1
                else:
                    label = -1 #we set negative label to -1
                #print (numerical_features)
                #print categorical_features
                j.write( "%s '%s %s\n" % (label,line[0],categorical_features))

            else: #we dont care about labels
                #print ( "1 '%s |i%s |c%s\n" % (line[0],numerical_features,categorical_features) )
                j.write( "1 '%s %s\n" % (line[0],categorical_features) )

  #Reporting progress
            #print counter
            if counter % 1000000 == 0:
                print("%s\t%s"%(counter, str(datetime.now() - start)))

    print("\n %s Task execution time:\n\t%s"%(counter, str(datetime.now() - start)))

#csv_to_vw("/Users/RahulAgarwal/kaggle_cpr/train", "/Users/RahulAgarwal/kaggle_cpr/click.train_original_data.vw",train=True)
#csv_to_vw("/Users/RahulAgarwal/kaggle_cpr/test", "/Users/RahulAgarwal/kaggle_cpr/click.test_original_data.vw",train=False)
```


## Running Vowpal Wabbit on the data

The Vowpal Wabbit will be run on the command line itself.

Training VW:
```bash
vw click.train_original_data.vw -f click.model.vw --loss_function logistic
```

Testing VW:

```bash
vw click.test_original_data.vw  -t -i click.model.vw -p click.preds.txt
```

## Creating Kaggle Submission File

```py
import math

def zygmoid(x):
    return 1 / (1 + math.exp(-x))

with open("kaggle.click.submission.csv","wb") as outfile:
    outfile.write("id,click\n")
    for line in open("click.preds.txt"):

        row = line.strip().split(" ")
        try:
            outfile.write("%s,%f\n"%(row[1],zygmoid(float(row[0]))))
        except:
            pass
```

This solution ranked 211/371 submissions at the time and the leaderboard score was 0.4031825 while the best leaderboard score was 0.3901120

## Next Steps

- Create a better VW model
    - Shuffle the data before making the model as the VW algorithm is an online learner and might have given more preference to the latest data
    - provide high weights for clicks as data is skewed. How Much?
    - tune VW algorithm using vw-hypersearch. What should be tuned?
    - Use categorical features like |C1 "C1"&"1"

- Create a XGBoost Model.
- Create a Sofia-ML Model and see how it works on this data.


<script src="//z-na.amazon-adsystem.com/widgets/onejs?MarketPlace=US&adInstanceId=c4ca54df-6d53-4362-92c0-13cb9977639e"></script>
