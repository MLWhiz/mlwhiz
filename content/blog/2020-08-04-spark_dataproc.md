---
title:  Accelerating Spark 3.0 Google DataProc Project with NVIDIA GPUs in 6 simple steps
date:  2020-08-04
draft: false
url : blog/2020/08/04/spark_dataproc/
slug: spark_dataproc
Category: Python

Keywords:
- Pandas
- Statistics

Categories:
- Big Data
- Data Science

Tags:
- Spark
- Machine Learning
- Data Science
- Artificial Intelligence
- Production

description:

thumbnail : /images/spark_dataproc/main.png
image : /images/spark_dataproc/main.png
toc : false
type: post
---


Data Exploration is a key part of Data Science. And does it take long? Ahh. Don’t even ask. Preparing a data set for ML not only requires understanding the data set, cleaning, and creating new features, it also involves doing these steps repeatedly until we have a fine-tuned system.

As we moved towards bigger datasets, [Apache Spark](https://towardsdatascience.com/the-hitchhikers-guide-to-handle-big-data-using-spark-90b9be0fe89a) came as a ray of hope. It gave us a scalable and distributed in-memory system to work with Big Data. By the by, we also saw frameworks like [Pytorch](https://towardsdatascience.com/moving-from-keras-to-pytorch-f0d4fff4ce79) and Tensorflow that inherently parallelized matrix computations using thousands of GPU cores.

But never did we see these two systems working in tandem in the past. We continued to use Spark for Big Data ETL tasks and GPUs for matrix intensive problems in [Deep Learning](https://towardsdatascience.com/stop-worrying-and-create-your-deep-learning-server-in-30-minutes-bb5bd956b8de).

![[Source](https://marketing.thepoweroftwo.solutions/acton/attachment/42621/f-0f867d92-c1d0-4112-afef-26f9cbb51499/1/-/-/-/-/NVIDIA%20&%20Google%20Cloud%20Dataproc%20ebook%20July2020.pdf)](/images/spark_dataproc/0.png)

And that is where Spark 3.0 comes. It provides us with a way to add NVIDIA GPUs to our Spark cluster nodes. The work done by these nodes can now be parallelized using both the CPU+GPU using the software platform for GPU computing, [RAPIDS](https://towardsdatascience.com/minimal-pandas-subset-for-data-scientist-on-gpu-d9a6c7759c7f?source=post_stats_page---------------------------).

> # Spark + GPU + RAPIDS = Spark 3.0

As per [NVIDIA](https://www.nvidia.com/en-in/deep-learning-ai/solutions/data-science/apache-spark-3/), the early adopters of Spark 3.0 already see a significantly faster performance with their current data loads. Such reductions in processing times can allow Data Scientists to perform more iterations on much bigger datasets, allowing Retailers to improve their forecasting, finance companies to enhance their credit models, and ad tech firms to improve their ability to predict click-through rates.

Excited yet. So how can you start using Spark 3.0? Luckily, Google Cloud, Spark, and NVIDIA have come together and simplified the cluster creation process for us. With Dataproc on Google Cloud, we can have a fully-managed Apache Spark cluster with GPUs in a few minutes.

***This post is about setting up your own Dataproc Spark Cluster with NVIDIA GPUs on Google Cloud.***

---

### 1. Create a New GCP Project

After the initial signup on the [Google Cloud Platform](https://cloud.google.com/), we can start a new project. Here I begin by creating a new project namedSparkDataProc.

![Create a New Project](/images/spark_dataproc/1.png)*Create a New Project*

---

### 2. Enable the APIs in the GCP Project

Once we add this project, we can go to our new project and start a Cloud Shell instance by clicking the “Activate **Cloud Shell**” button at the top right corner. Doing so will open up a terminal window at the bottom of our screen where we can run our next commands to set up a data proc cluster:

![Activate **Cloud Shell** for putting your commands](/images/spark_dataproc/2.png)

After this, we will need to run some commands to set up our project in the cloud shell. We start by enabling dataproc services within your project. Enable the Compute and Dataproc APIs to access Dataproc, and enable the Storage API as you’ll need a Google Cloud Storage bucket to house your data. We also set our default region. This may take several minutes:

    gcloud services enable compute.googleapis.com
    gcloud services enable dataproc.googleapis.com
    gcloud services enable storage-api.googleapis.com
    gcloud config set dataproc/region us-central1

---

### 3. Create and Put some data in GCS Bucket

Once done, we can create a new Google Cloud Storage Bucket, where we will keep all our data in the Cloud Shell:

    #You might need to change this name as this needs to be unique across all the users
    export BUCKET_NAME=rahulsparktest

    #Create the Bucket
    gsutil mb gs://${BUCKET_NAME}

We can also put some data in the bucket for later run purposes when we are running our spark cluster.

    # Get data in cloudshell terminal
    git clone https://github.com/caroljmcdonald/spark3-book
    mkdir -p ~/data/cal_housing
    tar -xzf spark3-book/data/cal_housing.tgz -C ~/data

    # Put data into Bucket using gsutil
    gsutil cp ~/data/CaliforniaHousing/cal_housing.data gs://${BUCKET_NAME}/data/cal_housing/cal_housing.csv

---

### 4. Setup the DataProc Rapids Cluster

To create a DataProc RAPIDS cluster that uses NVIDIA T4 GPUs, we need to get some initialization scripts that are used to instantiate our cluster. These scripts will install the GPU drivers([install_gpu_driver.sh](https://raw.githubusercontent.com/GoogleCloudDataproc/initialization-actions/master/gpu/install_gpu_driver.sh)) and create the Rapids conda environment([rapids.sh](https://raw.githubusercontent.com/GoogleCloudDataproc/initialization-actions/master/rapids/rapids.sh)) automatically for us. Since these scripts are in the development phase, the best way is to get the scripts from the GitHub source. We can do this using the below commands in our cloud shell in which we get the initialization scripts and copy them into our GS Bucket:

    wget https://raw.githubusercontent.com/GoogleCloudDataproc/initialization-actions/master/rapids/rapids.sh
    wget https://raw.githubusercontent.com/GoogleCloudDataproc/initialization-actions/master/gpu/install_gpu_driver.sh

    gsutil cp rapids.sh gs://$BUCKET_NAME
    gsutil cp install_gpu_driver.sh gs://$BUCKET_NAME

We can now create our cluster using the below command in the Cloud Shell. In the below command, we are using a predefined image version(2.0.0-RC2-ubuntu18) which has Spark 3.0 and python 3.7 to create our dataproc cluster. I am using a previous version of this image since the newest version has some issues with running Jupyter and Jupyter Lab. You can get a list of all versions [here](https://cloud.google.com/dataproc/docs/release-notes).

    CLUSTER_NAME=sparktestcluster
    REGION=us-central1
    gcloud beta dataproc clusters create ${CLUSTER_NAME} \
     --image-version 2.0.0-RC2-ubuntu18 \
     --master-machine-type n1-standard-8 \
     --worker-machine-type n1-highmem-32 \
     --worker-accelerator type=nvidia-tesla-t4,count=2 \
     --optional-components ANACONDA,JUPYTER,ZEPPELIN \
     --initialization-actions gs://$BUCKET_NAME/install_gpu_driver.sh,gs://$BUCKET_NAME/rapids.sh \
     --metadata rapids-runtime=SPARK \
     --metadata gpu-driver-provider=NVIDIA \
     --bucket ${BUCKET_NAME} \
     --subnet default \
     --enable-component-gateway \
    --properties="^#^spark:spark.task.resource.gpu.amount=0.125#spark:spark.executor.
    cores=8#spark:spark.task.cpus=1#spark:spark.yarn.unmanagedAM.enabled=false"

![**Cluster Architecture**](/images/spark_dataproc/3.png)

Our resulting Dataproc cluster has:

* One 8-core master node and two 32-core worker nodes

* Two NVIDIA T4 GPUs attached to each worker node

* Anaconda, Jupyter, and Zeppelin enabled

* Component gateway enabled for accessing Web UIs hosted on the cluster

* Extra Spark config tuning suitable for a notebook environment set using the properties flag. Specifically, we set spark.executor.cores=8 for improved parallelization and spark.yarn.unmanagedAM.enabled=false since it currently breaks ***SparkUI***.

***Troubleshooting:*** If you get errors regarding limits after this command, you might want to change some of the quotas in your default [Google Console Quotas Page](https://console.cloud.google.com/iam-admin/quotas?project=sparkdataproc). The limits I ended up changing were:

* **GPUs (all regions)** to 12 (Minimum:4)

* **CPUs (all regions)** to 164 (Minimum:72)

* **NVIDIA T4 GPUs** in us-central1 to 12 (Minimum:4)

* **CPUs** in us-central1 to 164 (Minimum:72)

I actually requested more limits than I required as the limit increase process might take a little longer and I will spin up some larger clusters later.

---

### 5. Run JupyterLab on DataProc Rapids Cluster

Once your command succeeds(It might take 10–15 mins) you will be able to see your Dataproc cluster at [https://console.cloud.google.com/dataproc/clusters](https://console.cloud.google.com/dataproc/clusters). Or you can go to the Google Cloud Platform console on your browser and search for “Dataproc” and click on the “Dataproc” icon(It looks like three connected circles). This will navigate you to the Dataproc clusters page.

![Dataproc Clusters Page](/images/spark_dataproc/4.png)

Now, you would be able to open a web interface(Jupyter/JupyterLab/Zeppelin) if you click on the sparktestcluster and then “Web Interfaces”.

![Web Interface Page for our Cluster](/images/spark_dataproc/5.png)

After opening up your Jupyter Pyspark Notebook, here is some example code for you to run if you are following along with this tutorial. In this code, we load a small dataset, and we see that the df.count() function ran in 252ms which is indeed fast for Spark, but I would do a much detailed benchmarking post later so keep tuned.

    file = "gs://rahulsparktest/data/cal_housing/cal_housing.csv"

    df = spark.read.load(file,format="csv", sep=",", inferSchema="true", header="false")
    colnames = ["longitude","latitude","medage","totalrooms","totalbdrms","population","houshlds","medincome","medhvalue"]

    df = df.toDF(*colnames)
    df.count()

![Yes Our **Jupyter Notebook** Works](/images/spark_dataproc/6.png)

### 6. Access the Spark UI

That is all well and done, but one major problem I faced was that I was not able to access the Spark UI using the link provided in the notebook. I found out that there were two ways to access the Spark UI for debugging purposes:

**A. Using the Web Interface option:**

We can access Spark UI by clicking first on **Yarn Resource Manager** Link on the **Web Interface** and then on Application Master on the corresponding page:

![Click on **Application Master** in **Tracking UI** Column to get Spark UI](/images/spark_dataproc/7.png)

And, you will arrive at the Spark UI Page:

![Yes! We get the Spark UI.](/images/spark_dataproc/8.png)

B. **Using the SSH Tunneling option:**

Another option to access the Spark UI is using Tunneling. To do this, you need to go to the Web Interface Page and click on *“Create an SSH tunnel to connect to a web interface”.*

![Spark Test Cluster Web Interface using SSH](/images/spark_dataproc/9.png)

This will give you two commands that you want to run on your **local machine** and not on Cloud shell. But before running them, you need to install google cloud SDK to your machine and set it up for your current project:

    sudo snap install google-cloud-sdk --classic

    # This Below command will open the browser where you can authenticate by selecting your own google account.
    gcloud auth login

    # Set up the project as sparkdataproc (project ID)
    gcloud config set project sparkdataproc

Once done with this, we can simply run the first command:

    gcloud compute ssh sparktestcluster-m --project=sparkdataproc  --zone=us-central1-b -- -D 1080 -N

And then the second one in another tab/window. This command will open up a new chrome window where you can access the Spark UI by clicking on Application Master the same as before.

    /usr/bin/google-chrome --proxy-server="socks5://localhost:1080"   --user-data-dir="/tmp/sparktestcluster-m" [http://sparktestcluster-m:8088](http://sparktestcluster-m:8088)

And that is it for setting up a Spark3.0 Cluster accelerated by GPUs.

It took me around 30 mins to go through all these steps if I don’t count the debugging time and the quota increase requests.


I am totally amazed by the concept of using a GPU on Spark and the different streams of experiments it opens up. Will be working on a lot of these in the coming weeks not only to benchmark but also because it is fun. So stay tuned.

---

## Continue Learning

Also, if you want to learn more about Spark and Spark DataFrames, I would like to call out an excellent course on [Big Data Essentials: HDFS, MapReduce and Spark RDD](https://coursera.pxf.io/4exq73) on Coursera.

I am going to be writing more of such posts in the future too. Let me know what you think about them. Follow me up at [Medium](https://mlwhiz.medium.com/) or Subscribe to my [blog](https://mlwhiz.ck.page/a9b8bda70c).
