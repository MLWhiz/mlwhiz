
# How I Created a Dataset for Instance Segmentation from Scratch?

How I Created a Dataset for Instance Segmentation from Scratch?

### The Simpsonssssssss…….

Recently, I was looking for a toy dataset for my new book’s chapter (you can subscribe to the updates [here](https://mlwhiz.ck.page/9a2ffe9e2c)) on instance segmentation. And, I really wanted to have something like the Iris Dataset for Instance Segmentation so that I would be able to explain the model without worrying about the dataset too much. But, alas, it is not always possible to get a dataset that you are looking for.

I actually ended up looking through various sources on the internet but inadvertently found that I would need to download a huge amount of data to get anything done. Given that is not at all the right way to go about any tutorial, I thought why not create my own dataset. 

In the end, it turned out to be the right decision as it was a fun task and as it provides an end to end perspective on what goes on in a real-world image detection/segmentation project.

This post is about creating your own custom dataset for Image Segmentation/Object Detection. 

## So, Why Not OID or any other dataset on the internet?

Though I was able to find out many datasets for Object Detection and classification per se, finding a dataset for instance segmentation was really hard. For example, Getting the images and annotations from the Open image dataset meant that I had to download all the mask images for the whole OID dataset effectively making it a 60GB download. Something I didn’t want to do for a tutorial. 

So I got to create one myself with the next question being which images I would want to tag? As the annotation work is pretty manual and lackluster, I wanted to have something that would at least induce some humor for me. 

So I went with tagging the Simpson’s. 

## What I created? 

In the end, I created a [dataset](https://www.kaggle.com/mlwhiz/simpsons-main-characters)(currently open-sourced on Kaggle) which contains ***81 image segmentations each for the five Simpson’s main characters*** (Homer, Lisa, Bert, Marge, and Maggie). While I could have started with even downloading Simpson’s images myself from the internet, I used the existing [Simpsons Characters Image Dataset](https://www.kaggle.com/alexattia/the-simpsons-characters-dataset?select=simpsons_dataset) from Kaggle to start with to save some time. This particular Simpsons character dataset had a lot of images, so I selected the first 80 for each of these 5 characters to manually annotate masks. And it took me a day and a half from start to finish with just around 400 images but yeah I like the end result which is shown below. 

![Author Image: Annotations for Bert, Maggie, Homer, Marge, and Lisa(Clockwise) ](https://cdn-images-1.medium.com/max/2000/0*II_2cGi3dw0ZW6QP)*Author Image: Annotations for Bert, Maggie, Homer, Marge, and Lisa(Clockwise) *

### File Descriptions:

You can also make use of this dataset under the Creative Commons license. And here are the file descriptions in the dataset. 

1. img folder: Contains images that I selected (81 per character) for annotation from the simpsons_dataset directory in Simpsons Characters Data.

1. instances.json : Contains annotations for files in img folder in the COCO format.

1. test_images folder: Contains all the other images that I didn't annotate from the simpsons_dataset directory in Simpsons Characters Data. These could be used to test or review your final model. 

## The Process

I remember using VIA annotation tool to create custom datasets a while back. But in this post, I would be using Hyperlabel for my labeling tasks as it provides a  really good UI interface out of the box and is free. The process is essentially pretty simple, yet I will go through the details to make it easier for any beginner. Please note that I assume that you already have got the images that you want to label in a folder. Also, the folder structure doesn’t matter much as long as all images are in it but it would really be helpful in modeling steps if you keep it a little consistent. I had my data structure as:

    - SimpsonsMainCharacters
       - bart_simpson
            - image0001.jpg
            - image0002.jpg
       - homer_simpson
       - lisa_simpson
       - marge_simpson
       - maggie_simpson

### 1. Set up Hyperlabel

Hyperlabel is an annotation tool that lets you annotate images with both bounding boxes(For Object Detection) as well as polygons(For instance segmentation). I found it as one of the best tools to do annotation as it is fast and provides a good UI. Also, you can get it for [free](https://hyperlabel.com/) for MAC and Windows. The only downside is that it is not available for Linux, but I guess that would be fine for most of us. I for one annotated my images on a Windows system and then moved the dataset to my Linux Machine. Once you install Hyperlabel, you can start with:

* **Creating the Project**: You can simply start by clicking on “Create Project”

![Author Image: Create a Project](https://cdn-images-1.medium.com/max/4608/0*1FgFXDCRUWPj7ltQ.png)*Author Image: Create a Project*

* Set the Project Name and **connect to your images**. I use the Local storage option, but you can select S3, Or Google Cloud also.

![Author Image: Set up Data Sources](https://cdn-images-1.medium.com/max/2418/1*YshljoFQ8K1DF0Qwe1c4bg.png)*Author Image: Set up Data Sources*

* **Define your labels**. Note that I select Polygons for all my character classes.

![Author Image: Set up Labels](https://cdn-images-1.medium.com/max/2000/1*2MpdhshTnnohE7GIOvHKzQ.png)*Author Image: Set up Labels*

* **Start Labeling**- First, select the person in the left pane and start annotating the mask. You can see your progress in the bottom pane.

![Author Image: Start annotating](https://cdn-images-1.medium.com/max/2492/1*VIHgTCn9SYzG69yABQ_u4w.png)*Author Image: Start annotating*

* **Export Labels**: Once you are done with your dataset, You can export in a variety of formats. I would use the COCO format for Instance segmentation as it is the most known and used format for Instance Segmentation. Also, remember to check the box against “Include Image files in JSON export”. Once you are finished, you will have a dir named img with all the images you annotated and a instances.json file.

![Author Image: Export](https://cdn-images-1.medium.com/max/2000/1*ir7xo2KUuP2v6w5vwxC_ag.png)*Author Image: Export*

### On the Side: Why I used the COCO Format?

The term COCO(Common Objects In Context) actually refers to a [dataset ](https://cocodataset.org/#home)on the Internet that contains 330K images, 1.5 million object instances, and 80 object categories.

But in the image community, it has gotten pretty normal to call the format that the COCO dataset was prepared with also as COCO. And this COCO format is actually what lots of current image detection libraries work with. For Instance in Detectron2, which is an awesome library for Instance segmentation by Facebook, using our Simpsons COCO dataset is as simple as:

    from detectron2.data.datasets import register_coco_instances

    register_coco_instances("simpsons_dataset", {}, "instances.json", "path/to/image/dir")

Don’t worry if you don’t understand the above code as I will get back to it in my next post where I will explain the COCO format along with creating an instance segmentation model on this dataset. Right now, just understand that COCO is a good format because it plays well with a lot of advanced libraries for such tasks and thus makes our lives a lot easier while moving between different models. 

## Conclusion

Creating your own dataset might take a lot of time but it is nonetheless a rewarding task. You help the community by providing a good resource, you also are able to understand and work on the project end to end when you create your own datasets. 

For instance, I understood just by annotating all these characters that it would be hard for the model to differentiate between Lisa and Maggie as they just look so similar.  Also, I would be amazed if the model is able to make a good mask around Lisa, Maggie, or Bert’s Zigzag Hair. At least for me, it took a long time to annotate them.

In my next post, I aim to explain the COCO format along with creating an instance segmentation model using Detectron2 on this dataset. So stay tuned.

### Keep Learning

If you want to know more about various*** Object Detection techniques, motion estimation, object tracking in video, etc***., I would like to recommend this excellent course on [Deep Learning in Computer Vision](https://www.coursera.org/specializations/aml?siteID=lVarvwc5BD0-AqkGMb7JzoCMW0Np1uLfCA&utm_content=2&utm_medium=partners&utm_source=linkshare&utm_campaign=lVarvwc5BD0) in the [Advanced machine learning specialization](https://www.coursera.org/specializations/aml?siteID=lVarvwc5BD0-AqkGMb7JzoCMW0Np1uLfCA&utm_content=2&utm_medium=partners&utm_source=linkshare&utm_campaign=lVarvwc5BD0). If you wish to know more about how the object detection field has evolved over the years, you can also take a look at my last [post](https://towardsdatascience.com/a-hitchhikers-guide-to-object-detection-and-instance-segmentation-ac0146fe8e11) on Object detection.

Thanks for the read. I am going to be writing more beginner-friendly posts in the future too. Follow me up at [**Medium](https://medium.com/@rahul_agarwal)** or Subscribe to my [**blog](http://eepurl.com/dbQnuX)** to be informed about them. As always, I welcome feedback and constructive criticism and can be reached on Twitter [@mlwhiz](https://twitter.com/MLWhiz)
