---
title:  Deployment could be easy — A Data Scientist’s Guide to deploy an Image detection FastAPI API using Amazon ec2
date:  2020-08-04
draft: false
url : blog/2020/08/08/deployment_fastapi/
slug: deployment_fastapi
Category: Python

Keywords:
- Pandas
- Statistics

Categories:
- Programming
- Computer Vision
- Awesome Guides

Tags:
- Machine Learning
- Data Science
- Production
- Productivity
- Tools
- Artificial Intelligence
- Computer Vision

description:

thumbnail : /images/deployment_fastapi/main.png
image : /images/deployment_fastapi/main.png
toc : false
type : post
---

Just recently, I had written a simple [tutorial](https://towardsdatascience.com/a-layman-guide-for-data-scientists-to-create-apis-in-minutes-31e6f451cd2f) on FastAPI, which was about simplifying and understanding how APIs work, and creating a simple API using the framework.

That post got quite a good response, but the most asked question was how to deploy the FastAPI API on ec2 and how to use images data rather than simple strings, integers, and floats as input to the API.

I scoured the net for this, but all I could find was some undercooked documentation and a lot of different ways people were taking to deploy using NGINX or ECS. None of those seemed particularly great or complete to me.

So, I tried to do this myself using some help from [FastAPI documentation](https://fastapi.tiangolo.com/deployment/). In this post, we will look at predominantly four things:

* Setting Up an Amazon Instance

* Creating a FastAPI API for Object Detection

* Deploying FastAPI using Docker

* An End to End App with UI

***So, without further ado, let’s get started.***

You can skip any part you feel you are versed with though I would expect you to go through the whole post, long as it may be, as there’s a lot of interconnection between concepts.

---
## 1. Setting Up Amazon Instance

Before we start with using the Amazon ec2 instance, we need to set one up. You might need to sign up with your email ID and set up the payment information on the [AWS website](https://aws.amazon.com/). Works just like a single sign-on. From here, I will assume that you have an AWS account and so I am going to explain the next essential parts so you can follow through.

* Go to AWS Management Console using [https://us-west-2.console.aws.amazon.com/console](https://us-west-2.console.aws.amazon.com/console).

* On the AWS Management Console, you can select “Launch a Virtual Machine.” Here we are trying to set up the machine where we will deploy our FastAPI API.

* In the first step, you need to choose the AMI template for the machine. I am selecting the 18.04 Ubuntu Server since Ubuntu.

![](/images/deployment_fastapi/0.png)

* In the second step, I select the t2.xlarge machine, which has 4 CPUs and 16GB RAM rather than the free tier since I want to use an Object Detection model and will need some resources.

![](/images/deployment_fastapi/1.png)

* Keep pressing Next until you reach the “6. Configure Security Group” tab. This is the most crucial step here. You will need to add a rule with Type: “HTTP” and Port Range:80.

![](/images/deployment_fastapi/2.png)

* You can click on “Review and Launch” and finally on the “Launch” button to launch the instance. Once you click on Launch, you might need to create a new key pair. Here I am creating a new key pair named fastapi and downloading that using the “Download Key Pair” button. Keep this key safe as it would be required every time you need to login to this particular machine. Click on “Launch Instance” after downloading the key pair

![](/images/deployment_fastapi/3.png)

* You can now go to your instances to see if your instance has started. Hint: See the Instance state; it should be showing “Running.”

![](/images/deployment_fastapi/4.png)

* Also, to note here are the Public DNS(IPv4) address and the IPv4 public IP. We will need it to connect to this machine. For me, they are:

```
Public DNS (IPv4): ec2-18-237-28-174.us-west-2.compute.amazonaws.com

IPv4 Public IP: 18.237.28.174
```
* Once you have that run the following commands in the folder, you saved the fastapi.pem file. If the file is named fastapi.txt you might need to rename it to fastapi.pem.

```
# run fist command if fastapi.txt gets downloaded.
# mv fastapi.txt fastapi.pem

chmod 400 fastapi.pem
ssh -i "fastapi.pem" ubuntu@<Your Public DNS(IPv4) Address>
```

![](/images/deployment_fastapi/5.png)

Now we have got our Amazon instance up and running. We can move on here to the real part of the post.

---
## 2. Creating a FastAPI API for Object Detection

Before we deploy an API, we need to have an API with us, right? In one of my last posts, I had written a simple [tutorial to understand FastAPI](https://towardsdatascience.com/a-layman-guide-for-data-scientists-to-create-apis-in-minutes-31e6f451cd2f) and API basics. Do read the post if you want to understand FastAPI basics.

So, here I will try to create an Image detection API. As for how to pass the Image data to the API? The idea is — ***What is an image but a string?*** An image is just made up of bytes, and we can encode these bytes as a string. We will use the base64 string representation, which is a popular way to get binary data to ASCII characters. And, we will pass this string representation to give an image to our API.

### A. Some Image Basics: What is Image, But a String?

So, let us first see how we can convert an Image to a String. We read the binary data from an image file using the ‘rb’ flag and turn it into a base64 encoded data representation using the base64.b64encode function. We then use the decode to utf-8 function to get the base encoded data into human-readable characters. Don’t worry if it doesn’t make a lot of sense right now. ***Just understand that any data is binary, and we can convert binary data to its string representation using a series of steps.***

As a simple example, if I have a simple image like below, we can convert it to a string using:

![dog_with_ball.jpg](/images/deployment_fastapi/6.png)
```py
import base64

with open("sample_images/dog_with_ball.jpg", "rb") as image_file:
    base64str = base64.b64encode(image_file.read()).decode("utf-8")
```

![We can get a string representation of any image](/images/deployment_fastapi/7.png)

Here I have got a string representation of a file named dog_with_ball.png on my laptop.

Great, we now have a string representation of an image. And, we can send this string representation to our FastAPI. But we also need to have a way to read an image back from its string representation. After all, our image detection API using PyTorch and any other package needs to have an image object that they can predict, and those methods don’t work on a string.

So here is a way to create a PIL image back from an image’s base64 string. Mostly we just do the reverse steps in the same order. We encode in ‘utf-8’ using .encode. We then use base64.b64decode to decode to bytes. We use these bytes to create a bytes object using io.BytesIO and use Image.open to open this bytes IO object as a PIL image, which can easily be used as an input to my PyTorch prediction code.*** Again simply, it is just a way to convert base64 image string to an actual image.***

```py
import base64
import io
from PIL import Image

def base64str_to_PILImage(base64str):
   base64_img_bytes = base64str.encode('utf-8')
   base64bytes = base64.b64decode(base64_img_bytes)
   bytesObj = io.BytesIO(base64bytes)
   img = Image.open(bytesObj)
   return img
```

So does this function work? Let’s see for ourselves. We can use just the string to get back the image.

![](/images/deployment_fastapi/8.png)

And we have our happy dog back again. Looks better than the string.

### B. Writing the Actual FastAPI code

So, as now we understand that our API can get an image as a string from our user, let’s create an object detection API that makes use of this image as a string and outputs the bounding boxes for the object with the object classes as well.

Here, I will be using a Pytorch pre-trained fasterrcnn_resnet50_fpn detection model from the torchvision.models for object detection, which is trained on the COCO dataset to keep the code simple, but one can use any model. You can look at these posts if you want to train your custom [image classification](https://towardsdatascience.com/end-to-end-pipeline-for-setting-up-multiclass-image-classification-for-data-scientists-2e051081d41c) or [image detection](https://lionbridge.ai/articles/create-an-end-to-end-object-detection-pipeline-using-yolov5/) model using Pytorch.

Below is the full code for the FastAPI. Although it may look long, we already know all the parts. In this code, we essentially do the following steps:

* Create our fast API app using the FastAPI() constructor.

* Load our model and the classes it was trained on. I got the list of classes from the PyTorch [docs](https://pytorch.org/docs/stable/torchvision/models.html).

* We also defined a new class Input , which uses a library called pydantic to validate the input data types that we will get from the API end-user. Here the end-user gives the base64str and some score threshold for object detection prediction.

* We add a function called base64str_to_PILImage which does just what it is named.

* And we write a predict function called get_predictionbase64 which returns a dict of bounding boxes and classes using a base64 string representation of an image and a threshold as an input. We also add [@app](http://twitter.com/app).put(“/predict”) on top of this function to define our endpoint. If you need to understand put and endpoint refer to my previous [post](https://towardsdatascience.com/a-layman-guide-for-data-scientists-to-create-apis-in-minutes-31e6f451cd2f) on FastAPI.

```py
from fastapi import FastAPI
from pydantic import BaseModel
import torchvision
from torchvision import transforms
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import numpy as np
import cv2
import io, json
import base64


app = FastAPI()

# load a pre-trained Model and convert it to eval mode.
# This model loads just once when we start the API.
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
model.eval()

# define the Input class
class Input(BaseModel):
    base64str : str
    threshold : float

def base64str_to_PILImage(base64str):
    base64_img_bytes = base64str.encode('utf-8')
    base64bytes = base64.b64decode(base64_img_bytes)
    bytesObj = io.BytesIO(base64bytes)
    img = Image.open(bytesObj)
    return img

@app.put("/predict")
def get_predictionbase64(d:Input):
    '''
    FastAPI API will take a base 64 image as input and return a json object
    '''
    # Load the image
    img = base64str_to_PILImage(d.base64str)
    # Convert image to tensor
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img)
    # get prediction on image
    pred = model([img])
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(float(i[0]), float(i[1])), (float(i[2]), float(i[3]))] for i in list(pred[0]['boxes'].detach().numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > d.threshold][-1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return {'boxes': pred_boxes,
        'classes' : pred_class}
```

### C. Local Before Global: Test the FastAPI code locally

Before we move on to AWS, let us check if the code works on our local machine. We can start the API on our laptop using:

    uvicorn fastapiapp:app --reload

The above means that your API is now running on your local server, and the --reload flag indicates that the API gets updated automatically when you change the fastapiapp.py file. This is very helpful while developing and testing, but you should remove this --reload flag when you put the API in production.

You should see something like:

![](/images/deployment_fastapi/9.png)

You can now try to access this API and see if it works using the requests module:

```py
import requests,json

payload = json.dumps({
  "base64str": base64str,
  "threshold": 0.5
})

response = requests.put("[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)",data = payload)
data_dict = response.json()
```

![](/images/deployment_fastapi/10.png)

And so we get our results using the API. This image contains a dog and a sports ball. We also have corner 1 (x1,y1) and corner 2 (x2,y2) coordinates of our bounding boxes.

### D. Lets Visualize

Although not strictly necessary, we can visualize how the results look in our Jupyter notebook:

```py

from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

def PILImage_to_cv2(img):
    return np.asarray(img)

def drawboundingbox(img, boxes,pred_cls, rect_th=2, text_size=1, text_th=2):
    img = PILImage_to_cv2(img)
    class_color_dict = {}

    #initialize some random colors for each class for better looking bounding boxes
    for cat in pred_cls:
        class_color_dict[cat] = [random.randint(0, 255) for _ in range(3)]

    for i in range(len(boxes)):
        cv2.rectangle(img, (int(boxes[i][0][0]), int(boxes[i][0][1])),
                      (int(boxes[i][1][0]),int(boxes[i][1][1])),
                      color=class_color_dict[pred_cls[i]], thickness=rect_th)
        cv2.putText(img,pred_cls[i], (int(boxes[i][0][0]), int(boxes[i][0][1])),  cv2.FONT_HERSHEY_SIMPLEX, text_size, class_color_dict[pred_cls[i]],thickness=text_th) # Write the prediction class
    plt.figure(figsize=(20,30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

img = Image.open("sample_images/dog_with_ball.jpg")
drawboundingbox(img, data_dict['boxes'], data_dict['classes'])
```

Here is the output:

![](/images/deployment_fastapi/11.png)

Here you will note that I got the image from the local file system, and that sort of can be considered as cheating as we don’t want to save every file that the user sends to us through a web UI. We should have been able to use the same base64string object that we also had to create this image. Right?

Not to worry, we could do that too. Remember our base64str_to_PILImage function? We could have used that also.

    img = base64str_to_PILImage(base64str)
    drawboundingbox(img, data_dict['boxes'], data_dict['classes'])

![](/images/deployment_fastapi/12.png)

That looks great. We have our working FastAPI, and we also have our amazon instance. We can now move on to Deployment.

---
## 3. Deployment on Amazon ec2

Till now, we have created an AWS instance and, we have also created a FastAPI that takes as input a base64 string representation of an image and returns bounding boxes and the associated class. But all the FastAPI code still resides in our local machine. ***How do we put it on the ec2 server? And run predictions on the cloud.***

### A. Install Docker

We will deploy our app using docker, as is suggested by the fastAPI creator himself. I will try to explain how docker works as we go. The below part may look daunting but it just is a series of commands and steps. So stay with me.

We can start by installing docker using:

    sudo apt-get update
    sudo apt install docker.io

We then start the docker service using:

    sudo service docker start

### B. Creating the folder structure for docker

    └── dockerfastapi
        ├── Dockerfile
        ├── app
        │   └── main.py
        └── requirements.txt

Here dockerfastapi is our project’s main folder. And here are the different files in this folder:

**i. requirements.txt:** Docker needs a file, which tells it which all libraries are required for our app to run. Here I have listed all the libraries I used in my Fastapi API.

    numpy
    opencv-python
    matplotlib
    torchvision
    torch
    fastapi
    pydantic

**ii. Dockerfile:** The second file is Dockerfile.

    FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

    COPY ./app /app
    COPY requirements.txt .
    RUN pip --no-cache-dir install -r requirements.txt

***How Docker works?:*** You can skip this section, but it will help to get some understanding of how docker works.

![](/images/deployment_fastapi/13.png)

The dockerfile can be thought of something like a sh file,which contains commands to create a docker image that can be run in a container. One can think of a docker image as an environment where everything like Python and Python libraries is installed. A container is a unit which is just an isolated box in our system that uses a dockerimage. The advantage of using docker is that we can create multiple docker images and use them in multiple containers. For example, one image might contain python36, and another can contain python37. And we can spawn multiple containers in a single Linux server.

Our Dockerfile contains a few things:

* FROM command: Here the first line FROM specifies that we start with tiangolo’s (FastAPI creator) Docker image. As per his site: “*This image has an “auto-tuning” mechanism included so that you can just add your code and get that same high performance automatically. And without making sacrifices”*. What we are doing is just starting from an image that installs python3.7 for us along with some added configurations for uvicorn and gunicorn ASGI servers and a start.sh file for ASGI servers automatically. For adventurous souls, particularly [commandset](https://github.com/tiangolo/uvicorn-gunicorn-fastapi-docker/blob/master/docker-images/python3.7.dockerfile)1 and [commandset2](https://github.com/tiangolo/uvicorn-gunicorn-docker/blob/master/docker-images/python3.7.dockerfile) get executed through a sort of a daisy-chaining of commands.

* COPY command: We can think of a docker image also as a folder that contains files and such. Here we copy our app folder and the requirements.txt file, which we created earlier to our docker image.

* RUN Command: We run pip install command to install all our python dependencies using the requirements.txt file that is now on the docker image.

**iii. main.py:** This file contains the fastapiapp.py code we created earlier. Remember to keep the name of the file main.py only.

### C. Docker Build

We have got all our files in the required structure, but we haven’t yet used any docker command. We will first need to build an image containing all dependencies using Dockerfile.

We can do this simply by:

    sudo docker build -t myimage .

This downloads, copies and installs some files and libraries from tiangolo’s image and creates an image called myimage. This myimage has python37 and some python packages as specified by requirements.txt file.

![](/images/deployment_fastapi/14.png)

We will then just need to start a container that runs this image. We can do this using:

    sudo docker run -d --name mycontainer -p 80:80 myimage

This will create a container named mycontainer which runs our docker image myimage. The part 80:80 connects our docker container port 80 to our Linux machine port 80.

![](/images/deployment_fastapi/15.png)

And actually that’s it. At this point, you should be able to open the below URL in your browser.

    # <IPV4 public IP>/docs
    URL: 18.237.28.174/docs

![](/images/deployment_fastapi/16.png)

And we can check our app programmatically using:

```py
payload = json.dumps({
  "base64str": base64str,
  "threshold": 0.5
})

response = requests.put("[http://18.237.28.174/predict](http://18.237.28.174/predict)",data = payload)
data_dict = response.json()
print(data_dict)

```

![](/images/deployment_fastapi/17.png)
> # Yup, finally our API is deployed.

### D. Troubleshooting as the real world is not perfect

All the above was good and will just work out of the box if you follow the exact instructions, but the real world doesn’t work like that. You will surely get some errors along the way and would need to debug your code. So to help you with that, some docker commands may come handy:

- **Logs:** When we ran our container using sudo docker run we don’t get a lot of info, and that is a big problem when you are debugging. You can see the real-time logs using the below command. If you see an error here, you will need to change your code and build the image again.

```
    sudo docker logs -f mycontainer
```

![](/images/deployment_fastapi/18.png)

- **Starting and Stopping Docker:** Sometimes, it might help just to restart your docker. In that case, you can use:

```
    sudo service docker stop
    sudo service docker start
```

* **Listing images and containers:** Working with docker, you will end up creating images and containers, but you won’t be able to see them in the working directory. You can list your images and containers using:

```
    sudo docker container ls
    sudo docker image ls
```

![](/images/deployment_fastapi/19.png)

* **Deleting unused docker images or containers:** You might need to remove some images or containers as these take up a lot of space on the system. Here is how you do that.

```
    # the prune command removes the unused containers and images
    sudo docker system prune

    # delete a particular container
    sudo docker rm mycontainer

    # remove myimage
    sudo docker image rm myimage

    # remove all images
    sudo docker image prune — all
```

* **Checking localhost:**The Linux server doesn’t have a browser, but we can still see the browser output though it’s a little ugly:

```
    curl localhost
```
![](/images/deployment_fastapi/20.png)

* **Develop without reloading image again and again:** For development, it’s useful to be able just to change the contents of the code on our machine and test it live, without having to build the image every time. In that case, it’s also useful to run the server with live auto-reload automatically at every code change. Here, we use our app directory on our Linux machine, and we replace the default (/start.sh) with the development alternative /start-reload.sh during development. After everything looks fine, we can build our image again run it inside the container.

```
    sudo docker run -d -p 80:80 -v $(pwd):/app myimage /start-reload.sh
```

If this doesn’t seem sufficient, adding here a docker cheat sheet containing useful docker commands:

![[Source](http://dockerlabs.collabnix.com/docker/cheatsheet/)](/images/deployment_fastapi/21.png)

---
## 4. An End to End App with UI

We are done here with our API creation, but we can also create a UI based app using [Streamlit](https://towardsdatascience.com/how-to-write-web-apps-using-simple-python-for-data-scientists-a227a1a01582) using our FastAPI API. This is not how you will do it in a production setting (where you might have developers making apps using react, node.js or javascript)but is mostly here to check the end-to-end flow of how to use an image API. I will host this barebones Streamlit app on local rather than the ec2 server, and it will get the bounding box info and classes from the FastAPI API hosted on ec2.

If you need to learn more about how streamlit works, you can check out this [post](https://towardsdatascience.com/how-to-write-web-apps-using-simple-python-for-data-scientists-a227a1a01582). Also, if you would want to deploy this streamlit app also to ec2, here is a [tutorial](https://towardsdatascience.com/how-to-deploy-a-streamlit-app-using-an-amazon-free-ec2-instance-416a41f69dc3) again.

Here is the flow of the whole app with UI and FastAPI API on ec2:

![Project Architecture](/images/deployment_fastapi/22.png)*Project Architecture*

The most important problems we need to solve in our streamlit app are:

### How to get an image file from the user using Streamlit?

**A. Using File uploader:** We can use the file uploader using:

```py
bytesObj = st.file_uploader(“Choose an image file”)
```

The next problem is, what is this bytesObj we get from the streamlit file uploader? In streamlit, we will get a bytesIO object from the file_uploader and we will need to convert it to base64str for our FastAPI app input. This can be done using:

```py
def bytesioObj_to_base64str(bytesObj):
   return base64.b64encode(bytesObj.read()).decode("utf-8")

base64str = bytesioObj_to_base64str(bytesObj)
```
**B. Using URL:** We can also get an image URL from the user using text_input.

```py
url = st.text_input(‘Enter URL’)
```
We can then get image from URL in base64 string format using the requests module and base64 encode and utf-8 decode:

```py
def ImgURL_to_base64str(url):
    return base64.b64encode(requests.get(url).content).decode("utf-8")

base64str = ImgURL_to_base64str(url)
```

And here is the complete code of our Streamlit app. You have seen most of the code in this post already.

```py
import streamlit as st
import base64
import io
import requests,json
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
import random

# use file uploader object to recieve image
# Remember that this bytes object can be used only once
def bytesioObj_to_base64str(bytesObj):
    return base64.b64encode(bytesObj.read()).decode("utf-8")

# Image conversion functions

def base64str_to_PILImage(base64str):
    base64_img_bytes = base64str.encode('utf-8')
    base64bytes = base64.b64decode(base64_img_bytes)
    bytesObj = io.BytesIO(base64bytes)
    img = Image.open(bytesObj)
    return img

def PILImage_to_cv2(img):
    return np.asarray(img)

def ImgURL_to_base64str(url):
    return base64.b64encode(requests.get(url).content).decode("utf-8")

def drawboundingbox(img, boxes,pred_cls, rect_th=2, text_size=1, text_th=2):
    img = PILImage_to_cv2(img)
    class_color_dict = {}

    #initialize some random colors for each class for better looking bounding boxes
    for cat in pred_cls:
        class_color_dict[cat] = [random.randint(0, 255) for _ in range(3)]

    for i in range(len(boxes)):
        cv2.rectangle(img, (int(boxes[i][0][0]), int(boxes[i][0][1])),
                      (int(boxes[i][1][0]),int(boxes[i][1][1])),
                      color=class_color_dict[pred_cls[i]], thickness=rect_th)
        cv2.putText(img,pred_cls[i], (int(boxes[i][0][0]), int(boxes[i][0][1])),  cv2.FONT_HERSHEY_SIMPLEX, text_size, class_color_dict[pred_cls[i]],thickness=text_th)
    plt.figure(figsize=(20,30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

st.markdown("<h1>Our Object Detector App using FastAPI</h1><br>", unsafe_allow_html=True)

bytesObj = st.file_uploader("Choose an image file")

st.markdown("<center><h2>or</h2></center>", unsafe_allow_html=True)

url = st.text_input('Enter URL')

if bytesObj or url:
    # In streamlit we will get a bytesIO object from the file_uploader
    # and we convert it to base64str for our FastAPI
    if bytesObj:
        base64str = bytesioObj_to_base64str(bytesObj)

    elif url:
        base64str = ImgURL_to_base64str(url)

    # We will also create the image in PIL Image format using this base64 str
    # Will use this image to show in matplotlib in streamlit
    img = base64str_to_PILImage(base64str)

    # Run FastAPI
    payload = json.dumps({
      "base64str": base64str,
      "threshold": 0.5
    })

    response = requests.put("http://18.237.28.174/predict",data = payload)
    data_dict = response.json()


    st.markdown("<center><h1>App Result</h1></center>", unsafe_allow_html=True)
    drawboundingbox(img, data_dict['boxes'], data_dict['classes'])
    st.pyplot()
    st.markdown("<center><h1>FastAPI Response</h1></center><br>", unsafe_allow_html=True)
    st.write(data_dict)
```

We can run this streamlit app in local using:

    streamlit run streamlitapp.py

And we can see our app running on our localhost:8501. Works well with user-uploaded images as well as URL based images. Here is a cat image for some of you cat enthusiasts as well.

<table><tr><td><img src='/images/deployment_fastapi/23.png'></td><td><img src='/images/deployment_fastapi/24.png'></td></tr></table>

So that’s it. We have created a whole workflow here to deploy image detection models through FastAPI on ec2 and utilizing those results in Streamlit. I hope this helps your woes around deploying models in production. You can find the code for this post as well as all my posts at my [GitHub](https://github.com/MLWhiz/data_science_blogs/tree/master/deployFastApi) repository.

Let me know if you like this post and if you would like to include Docker or FastAPI or Streamlit in your day to day deployment needs. I am also looking to create a much detailed post on Docker so follow me up to stay tuned with my writing as well. Details below.

---
## Continue Learning

If you want to learn more about building and putting a Machine Learning model in production, this [course on AWS](https://click.linksynergy.com/link?id=lVarvwc5BD0&offerid=467035.14884356434&type=2&murl=https%3A%2F%2Fwww.coursera.org%2Flearn%2Faws-machine-learning) for implementing Machine Learning applications promises just that.

Thanks for the read. I am going to be writing more beginner-friendly posts in the future too. Follow me up at [Medium](https://mlwhiz.medium.com/?source=post_page---------------------------) or Subscribe to my [blog](https://mlwhiz.ck.page/a9b8bda70c)

Also, a small disclaimer — There might be some affiliate links in this post to relevant resources, as sharing knowledge is never a bad idea.
