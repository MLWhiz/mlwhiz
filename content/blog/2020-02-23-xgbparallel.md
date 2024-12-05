---
title: Lightning Fast XGBoost on Multiple GPUs
date:  2020-02-23
draft: false
url : blog/2020/02/23/xgbparallel/
slug: xgbparallel
Category: Python

Keywords:
- Pandas
- Statistics

Categories:
- Data Science
- programming

Tags:
- Machine Learning
- Data Science
- Artificial Intelligence
- xgboost
- Dask
- Productivity

description: This post is about running XGBoost on Multi-GPU machines.

thumbnail : /images/xgbparallel/main.png
image : /images/xgbparallel/main.png
toc : false
type : post
---

XGBoost is one of the most used libraries fora data science.

At the time XGBoost came into existence, it was lightning fast compared to its nearest rival Python’s Scikit-learn GBM. But as the times have progressed, it has been rivaled by some awesome libraries like LightGBM and Catboost, both on speed as well as accuracy.

I, for one, use LightGBM for most of the use cases where I have just got CPU for training. But when I have a GPU or multiple GPUs at my disposal, I still love to train with XGBoost.

Why?

So I could make use of the excellent GPU Capabilities provided by XGBoost in conjunction with Dask to use XGBoost in both single and multi-GPU mode.

How?

***This post is about running XGBoost on Multi-GPU machines.***

---

## Dataset:

![UCI Higgs](/images/xgbparallel/0.png)

We are going to be using the [UCI Higgs dataset](https://archive.ics.uci.edu/ml/datasets/HIGGS). This is a binary classification problem with 11M rows and 29 columns and can take a considerable time to solve.

From the UCI Site:

> The data has been produced using Monte Carlo simulations. The first 21 features (columns 2–22) are kinematic properties measured by the particle detectors in the accelerator. The last seven features are functions of the first 21 features; these are high-level features derived by physicists to help discriminate between the two classes. There is an interest in using deep learning methods to obviate the need for physicists to manually develop such features. Benchmark results using Bayesian Decision Trees from a standard physics package and 5-layer neural networks are presented in the original paper. The last 500,000 examples are used as a test set.

We can load this dataset into memory by using the nifty function that I borrow from [this NVidia post](https://devblogs.nvidia.com/gradient-boosting-decision-trees-xgboost-cuda/).

```py
if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve

data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
dmatrix_train_filename = "higgs_train.dmatrix"
dmatrix_test_filename = "higgs_test.dmatrix"
csv_filename = "HIGGS.csv.gz"
train_rows = 10500000
test_rows = 500000
num_round = 1000

plot = True

# return xgboost dmatrix
def load_higgs():
    if os.path.isfile(dmatrix_train_filename) and os.path.isfile(dmatrix_test_filename):
        dtrain = xgb.DMatrix(dmatrix_train_filename)
        dtest = xgb.DMatrix(dmatrix_test_filename)
        if dtrain.num_row() == train_rows and dtest.num_row() == test_rows:
            print("Loading cached dmatrix...")
            return dtrain, dtest

    if not os.path.isfile(csv_filename):
        print("Downloading higgs file...")
        urlretrieve(data_url, csv_filename)

    df_higgs_train = pandas.read_csv(csv_filename, dtype=np.float32,
                                     nrows=train_rows, header=None)
    dtrain = xgb.DMatrix(df_higgs_train.ix[:, 1:29], df_higgs_train[0])
    dtrain.save_binary(dmatrix_train_filename)
    df_higgs_test = pandas.read_csv(csv_filename, dtype=np.float32,
                                    skiprows=train_rows, nrows=test_rows,
                                    header=None)
    dtest = xgb.DMatrix(df_higgs_test.ix[:, 1:29], df_higgs_test[0])
    dtest.save_binary(dmatrix_test_filename)

    return dtrain, dtest

dtrain, dtest = load_higgs()
```

This function downloads the Higgs dataset and creates Dmatrix objects for later XGBoost use.

---

## XGBoost: The CPU Method

![[Source](https://pixabay.com/illustrations/processor-cpu-computer-chip-board-2217771/)](/images/xgbparallel/1.png)

As we have the data loaded, we can train the XGBoost model with CPU for benchmarking purposes.

    print("Training with CPU ...")
    param = {}
    param['objective'] = 'binary:logitraw'
    param['eval_metric'] = 'error'
    param['silent'] = 1
    param['tree_method'] = 'hist'

    tmp = time.time()
    cpu_res = {}
    xgb.train(param, dtrain, num_round, evals=[(dtest, "test")],
              evals_result=cpu_res)
    cpu_time = time.time() - tmp
    print("CPU Training Time: %s seconds" % (str(cpu_time)))

    ---------------------------------------------------------------

    CPU Training Time: 717.6483490467072 seconds

This code takes 717 seconds, which is around 12 minutes to finish. That is great and commendable, but can we do better?

---

## XGBoost: The Single GPU Method

![[Source](https://www.google.com/url?sa=i&rct=j&q=&esrc=s&source=images&cd=&cad=rja&uact=8&ved=2ahUKEwiKwMPphI3nAhXtzzgGHeLNB1QQjhx6BAgBEAI&url=https%3A%2F%2Fwww.nvidia.com%2Fen-us%2Fdeep-learning-ai%2Fproducts%2Ftitan-rtx%2F&psig=AOvVaw3QFPz_lTQrqraHJQz_ewwh&ust=1579433051897517)](/images/xgbparallel/2.png)

What is great is that we don’t have to change a lot in the above code to be able to use a single GPU for our model building.

> Why use CPU when we can use GPU?

We change the tree_method to gpu_hist

    print("Training with Single GPU ...")

    param = {}
    param['objective'] = 'binary:logitraw'
    param['eval_metric'] = 'error'
    param['silent'] = 1
    param['tree_method'] = 'gpu_hist'
    tmp = time.time()
    gpu_res = {}

    xgb.train(param, dtrain, num_round, evals=[(dtest, "test")],
              evals_result=gpu_res)
    gpu_time = time.time() - tmp
    print("GPU Training Time: %s seconds" % (str(gpu_time)))

    ----------------------------------------------------------------
    GPU Training Time: 78.2187008857727 seconds

And ***we achieve a 10x speedup*** with our model now finishing in 1.3 minutes. That is great, but can we do even better if we have multiple GPUs?

---

## XGBoost: The Multi GPU Method

![[Source](https://www.google.com/imgres?imgurl=https%3A%2F%2Fwww.nvidia.com%2Fcontent%2Fdam%2Fen-zz%2FSolutions%2Ftitan%2Ftitan-rtx%2Fnvidia-titan-rtx-nvlink-300-t%402x.jpg&imgrefurl=https%3A%2F%2Fwww.nvidia.com%2Fen-us%2Fdeep-learning-ai%2Fproducts%2Ftitan-rtx%2F&docid=LXUW4PiXbaOvoM&tbnid=2aobaeCTl1m0CM%3A&vet=10ahUKEwi2uovhhI3nAhXVcn0KHf9pB_YQMwhLKAIwAg..i&w=600&h=408&client=ubuntu&bih=2025&biw=1879&q=gpu%20titan%20rtx%20dual&ved=0ahUKEwi2uovhhI3nAhXVcn0KHf9pB_YQMwhLKAIwAg&iact=mrc&uact=8)](/images/xgbparallel/3.png)

I have, for example, 2 GPUs in my machine while the above code utilizes only 1 GPU. With GPU’s getting a lot cheaper now, it is not unusual for clusters to have more than 4 GPUs. So can we use multiple GPUs simultaneously?

> Two GPUs are always better than one

To use MultiGPUs, the process is not so simple as to add a little argument as above, and there are a few steps involved.

The first is the difference in Data loading:

```py
def load_higgs_for_dask(client):
    # 1. read the CSV File using Pandas
    df_higgs_train = pandas.read_csv(csv_filename, dtype=np.float32,
                                     nrows=train_rows, header=None).ix[:, 0:30]
    df_higgs_test = pandas.read_csv(csv_filename, dtype=np.float32,
                                    skiprows=train_rows, nrows=test_rows,
                                    header=None).ix[:, 0:30]

    # 2. Create a Dask Dataframe from Pandas Dataframe.
    ddf_higgs_train = dask.dataframe.from_pandas(df_higgs_train, npartitions=8)
    ddf_higgs_test = dask.dataframe.from_pandas(df_higgs_test, npartitions=8)
    ddf_y_train = ddf_higgs_train[0]
    del ddf_higgs_train[0]
    ddf_y_test = ddf_higgs_test[0]
    del ddf_higgs_test[0]

    #3. Create Dask DMatrix Object using dask dataframes
    ddtrain = DaskDMatrix(client, ddf_higgs_train ,ddf_y_train)
    ddtest = DaskDMatrix(client, ddf_higgs_test ,ddf_y_test)

    return ddtrain, ddtest
```

There are multiple steps in data load as we need dask DMatrix objects to train XGBoost with multiple GPUs.

1. Read the CSV File using Pandas.

1. Create a Dask Dataframe from Pandas Dataframe, and

1. Create Dask DMatrix Object using dask data frames.

To use Multi-GPU for training XGBoost, we need to use Dask to create a GPU Cluster. This command creates a cluster of our GPUs that could be used by dask by using the client object later.

    cluster = LocalCUDACluster()
    client = Client(cluster)

We can now load our Dask Dmatrix Objects and define the training parameters. Note nthread beings set to one and tree_method set to gpu_hist

    ddtrain, ddtest = load_higgs_for_dask(client)

    param = {}
    param['objective'] = 'binary:logitraw'
    param['eval_metric'] = 'error'
    param['silence'] = 1
    param['tree_method'] = 'gpu_hist'
    param['nthread'] = 1

We can now train on Multiple GPUs using:

    print("Training with Multiple GPUs ...")
    tmp = time.time()
    output = xgb.dask.train(client, param, ddtrain, num_boost_round=1000, evals=[(ddtest, 'test')])
    multigpu_time = time.time() - tmp
    bst = output['booster']
    multigpu_res = output['history']
    print("Multi GPU Training Time: %s seconds" % (str(multigpu_time)))
    ---------------------------------------------------------------
    Multi GPU Training Time: 50.08211898803711 seconds

Please note how the call to xgb.train changes to xgb.dask.train and how it also needs the dask client to work.

This took around 0.8 Minutes that is a 1.5x Speedup from Single GPU. I only had 2 GPUs at my disposal, so I can’t test it, but I believe that it increases linearly, i.e. more GPU and more reduction in time.

---

## Results

Here are the results of all three setups:

![](/images/xgbparallel/4.png)

Although the difference between Multi and Single CPU looks redundant right now, it will be pretty considerable while running [multiple hyperparameter tuning tasks](https://towardsdatascience.com/automate-hyperparameter-tuning-for-your-models-71b18f819604) at hand where one might need to run multiple GBM Models with different Hyperparams.

Also, this result can change when we scale it to many GPUs.

So keep scaling.

You can find the complete code for this post on [Github](https://github.com/MLWhiz/data_science_blogs/blob/master/xgb_dask/XGB%20with%20Dask%2BGPU.ipynb).

---

## Continue Learning

If you want to learn more about Deep Learning, I would like to recommend this awesome [Deep Learning Specialization](https://imp.i384100.net/LXZO13). You can start for free with the 7-day Free Trial. This specialization talks about various CNN architetures and covers a wide variety of problems in the image domain including detection and segmentation.

Thanks for the read. I am going to be writing more beginner-friendly posts in the future too. Follow me up at [**Medium**](https://mlwhiz.medium.com/?source=post_page---------------------------) or Subscribe to my [**blog**](mlwhiz.com)

Also, a small disclaimer — There might be some affiliate links in this post to relevant resources, as sharing knowledge is never a bad idea.
