---
title: "Object Detection: An End to End Theoretical Perspective"
date:  2018-09-22
draft: false
url : blog/2018/09/22/object_detection/
slug: object_detection
Category: deep learning, image
Keywords: 
- rcnn
- faster rcnn
- roi pooling
- image segmentation
- object detection
Tags: 
- object detection
- rcnn
- faster rcnn
description: A thorough walkthrough in the object detection space
toc : false
---

We all know about the image classification problem. Given an image can you find out the class the image belongs to? We can solve any new image classification problem with ConvNets and [Transfer Learning](https://medium.com/@14prakash/transfer-learning-using-keras-d804b2e04ef8) using pre-trained nets.
<br>
<div style="color:black; background-color: #E9DAEE;">
ConvNet as fixed feature extractor. Take a ConvNet pretrained on ImageNet, remove the last fully-connected layer (this layer's outputs are the 1000 class scores for a different task like ImageNet), then treat the rest of the ConvNet as a fixed feature extractor for the new dataset. In an AlexNet, this would compute a 4096-D vector for every image that contains the activations of the hidden layer immediately before the classifier. We call these features CNN codes. It is important for performance that these codes are ReLUd (i.e. thresholded at zero) if they were also thresholded during the training of the ConvNet on ImageNet (as is usually the case). Once you extract the 4096-D codes for all images, train a linear classifier (e.g. Linear SVM or Softmax classifier) for the new dataset.
</div>
<br>

As a side note: if you want to know more about convnets and Transfer Learning I would like to recommend this awesome course on [Deep Learning in Computer Vision](https://www.coursera.org/specializations/aml?siteID=lVarvwc5BD0-AqkGMb7JzoCMW0Np1uLfCA&utm_content=2&utm_medium=partners&utm_source=linkshare&utm_campaign=lVarvwc5BD0) in the [Advanced machine learning specialization](https://www.coursera.org/specializations/aml?siteID=lVarvwc5BD0-AqkGMb7JzoCMW0Np1uLfCA&utm_content=2&utm_medium=partners&utm_source=linkshare&utm_campaign=lVarvwc5BD0). You can start for free with the 7-day Free Trial. This course talks about various CNN architetures and covers a wide variety of problems in the image domain including detection and segmentation.

But there are a lot many interesting problems in the Image domain. The one which we are going to focus on today is the Segmentation, Localization and Detection problem.
So what are these problems?

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/id1.png"  height="400" width="700" ></center>
</div>

So these problems are divided into 4 major buckets. In the next few lines I would try to explain each of these problems concisely before we take a deeper dive:

1. Semantic Segmentation: Given an image, can we classify each pixel as belonging to a particular class?
2. Classification+Localization: We were able to classify an image as a cat. Great. Can we also get the location of the said cat in that image by drawing a bounding box around the cat? Here we assume that there is a fixed number(commonly 1) in the image.
3. Object Detection: A More general case of the Classification+Localization problem. In a real-world setting, we don't know how many objects are in the image beforehand. So can we detect all the objects in the image and draw bounding boxes around them?
4. Instance Segmentation: Can we create masks for each individual object in the image? It is different from semantic segmentation. How? If you look in the 4th image on the top, we won't be able to distinguish between the two dogs using semantic segmentation procedure as it would sort of merge both the dogs together.

In this post, we will focus mainly on Object Detection.

## Classification+Localization

So lets first try to understand how we can solve the problem when we have a single object in the image. The Classification+Localization case. Pretty neatly said in the CS231n notes:

<div style="color:black; background-color: #E9DAEE;">
Treat localization as a regression problem!
</div>

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/id2.png"  height="400" width="700" ></center>
</div>

**Input Data:** Lets first talk about what sort of data such sort of model expects. Normally in an image classification setting we used to have data in the form (X,y) where X is the image and y used to be the class labels.
In the Classification+Localization setting we will have data normally in the form (X,y), where X is still the image and y is a array containing (class_label, x,y,w,h) where,

x = bounding box top left corner x-coordinate

y = bounding box top left corner y-coordinate

w = width of bounding box in pixel

h = height of bounding box in pixel

**Model:** So in this setting we create a multi-output model which takes an image as the input and has (n_labels + 4) output nodes. n_labels nodes for each of the output class and 4 nodes that give the predictions for (x,y,w,h).

**Loss:** In such a setting setting up the loss is pretty important. Normally the loss is a weighted sum of the Softmax Loss(from the Classification Problem) and the regression L2 loss(from the bounding box coordinates).

$$Loss = alpha*SoftmaxLoss + (1-alpha)*L2Loss$$

Since these two losses would be on a different scale, the alpha hyper-parameter needs to be tuned.

*There is one thing I would like to note here. We are trying to do object localization task but we still have our convnets in place here. We are just adding one more output layer to also predict the coordinates of the bounding box and tweaking our loss function. And here in lies the essence of the whole Deep Learning framework - Stack layers on top of each other, reuse components to create better models, and create architectures to solve your own problem. And that is what we are going to see a lot going forward.*

## Object Detection

*So how does this idea of localization using regression get mapped to Object Detection?* It doesn't. We don't have a fixed number of objects. So we can't have 4 outputs denoting, the bounding box coordinates.

One naive idea could be to apply a CNN to many different crops of the image, CNN classifies each crop as object class or background class. This is intractable. There could be a lot of such crops that you can create.

### Region Proposals:

If just there was a method(Normally called Region Proposal Network)which could find some cropped regions for us automatically, we could just run our convnet on those regions and be done with object detection. And that is what selective search (Uijlings et al, "[Selective Search for Object Recognition](https://medium.com/r/?url=http%3A%2F%2Fwww.huppelen.nl%2Fpublications%2FselectiveSearchDraft.pdf)", IJCV 2013) provided for RCNN.

So what are Region Proposals:

- Find *"blobby"* image regions that are likely to contain objects
- Relatively fast to run; e.g. Selective Search gives 2000 region proposals in a few seconds on CPU

How the region proposals are being made?

### Selective Search for Object Recognition:

So this paper starts with a set of some initial regions using [13] (P. F. Felzenszwalb and D. P. Huttenlocher. [Efficient GraphBased Image Segmentation](https://medium.com/r/?url=http%3A%2F%2Fpeople.cs.uchicago.edu%2F~pff%2Fpapers%2Fseg-ijcv.pdf). IJCV, 59:167–181, 2004. 1, 3, 4, 5, 7)
<br>
<div style="color:black; background-color: #E9DAEE;">
Graph-based image segmentation techniques generally represent the problem in terms of a graph G = (V, E) where each node v ∈ V corresponds to a pixel in the image, and the edges in E connect certain pairs of neighboring pixels. A weight is associated with each edge based on some property of the pixels that it connects, such as their image intensities. Depending on the method, there may or may not be an edge connecting each pair of vertices.
</div>
<br>
In this paper they take an approach:
<br>
<div style="color:black; background-color: #E9DAEE;">
Each edge (vi , vj )∈ E has a corresponding weight w((vi , vj )), which is a non-negative measure of the dissimilarity between neighboring elements vi and vj . In the case of image segmentation, the elements in V are pixels and the weight of an edge is some measure of the dissimilarity between the two pixels connected by that edge (e.g., the difference in intensity, color, motion, location or some other local attribute). In the graph-based approach, a segmentation S is a partition of V into components such that each component (or region) C ∈ S corresponds to a connected component in a graph.
</div>
<br>
<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/id3.png"  height="400" width="700" ></center>
</div>

As you can see if we create bounding boxes around these masks we will be losing a lot of regions. We want to have the whole baseball player in a single bounding box/frame. We need to somehow group these initial regions.
For that the authors of [Selective Search for Object Recognition](https://medium.com/r/?url=http%3A%2F%2Fwww.huppelen.nl%2Fpublications%2FselectiveSearchDraft.pdf) apply the Hierarchical Grouping algorithm to these initial regions. In this algorithm they merge most similar regions together based on different notions of similarity based on colour, texture, size and fill.

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/id5.png"  height="400" width="700" ></center>
</div>

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/id6.png"  height="400" width="700" ></center>
</div>

## RCNN

The above selective search is the region proposal they used in RCNN paper. But what is RCNN and how does it use region proposals?

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/id7.png"  height="400" width="700" ></center>
</div>

<div style="color:black; background-color: #E9DAEE;">
Object detection system overview. Our system
(1) takes an input image, (2) extracts around 2000 bottom-up region proposals, (3) computes features for each proposal using a large convolutional neural network (CNN), and then (4) classifies each region using class-specific linear SVM.
</div>
<br>
Along with this, the authors have also used a class specific bounding box regressor, that takes:
Input : (Px,Py,Ph,Pw) - the location of the proposed region.
Target: (Gx,Gy,Gh,Gw) - Ground truth labels for the region.
The goal is to learn a transformation that maps the proposed region(P) to the Ground truth box(G)

### Training RCNN

What is the input to an RCNN?
So we have got an image, Region Proposals from the RPN strategy and the ground truths of the labels (labels, ground truth boxes)
Next we treat all region proposals with ≥ 0.5 IoU(Intersection over union) overlap with a ground-truth box as positive training example for that box's class and the rest as negative. We train class specific SVM's

So every region proposal becomes a training example. and the convnet gives a feature vector for that region proposal. We can then train our n-SVMs using the class specific data.

### Test Time RCNN

At test time we predict detection boxes using class specific SVMs. We will be getting a lot of overlapping detection boxes at the time of testing. Non-maximum suppression is an integral part of the object detection pipeline. First, it sorts all detection boxes on the basis of their scores. The detection box M with the maximum score is selected and all other detection boxes with a significant overlap (using a pre-defined threshold) with M are suppressed. This process is recursively applied on the remaining boxes

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/id8.jpeg"  height="400" width="700" ></center>
</div>

### Problems with RCNN:

Training is slow.
Inference (detection) is slow. 47s / image with VGG16 - Since the Convnet needs to be run many times.

Need for speed. Hence comes in picture by the same authors:

## Fast RCNN

<div style="color:black; background-color: #E9DAEE;">
So the next idea from the same authors: Why not create convolution map of input image and then just select the regions from that convolutional map? Do we really need to run so many convnets? What we can do is run just a single convnet and then apply region proposal crops on the features calculated by the convnet and use a simple SVM to classify those crops.
</div>
<br>
Something like:

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/id9.png"  height="400" width="700" ></center>
</div>

<div style="color:black; background-color: #E9DAEE;">
From Paper: Fig. illustrates the Fast R-CNN architecture. A Fast R-CNN network takes as input an entire image and a set of object proposals. The network first processes the whole image with several convolutional (conv) and max pooling layers to produce a conv feature map. Then, for each object proposal a region of interest (RoI) pooling layer extracts a fixed-length feature vector from the feature map. Each feature vector is fed into a sequence of fully connected (fc) layers that finally branch into two sibling output layers: one that produces softmax probability estimates over K object classes plus a catch-all "background" class and another layer that outputs four real-valued numbers for each of the K object classes. Each set of 4 values encodes refined bounding-box positions for one of the K classes.
</div>
<br>
This idea depends a little upon the architecture of the model that get used too. Do we take the 4096 bottleneck layer from VGG16?
So the architecture that the authors have proposed is:

<div style="color:black; background-color: #E9DAEE;">
We experiment with three pre-trained ImageNet [4] networks, each with five max pooling layers and between five and thirteen conv layers (see Section 4.1 for network details). When a pre-trained network initializes a Fast R-CNN network, it undergoes three transformations. First, the last max pooling layer is replaced by a RoI pooling layer that is configured by setting H and W to be compatible with the net's first fully connected layer (e.g., H = W = 7 for VGG16). Second, the network's last fully connected layer and softmax (which were trained for 1000-way ImageNet classification) are replaced with the two sibling layers described earlier (a fully connected layer and softmax over K + 1 categories and category-specific bounding-box regressors). Third, the network is modified to take two data inputs: a list of images and a list of RoIs in those images.
</div>
<br>
This obviously is a little confusing and "hairy", let us break this down. But for that, we need to see the VGG16 architecture.

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/id10.png"  height="400" width="700" ></center>
</div>

The last pooling layer is 7x7x512. This is the layer the network authors intend to replace by the ROI pooling layers. This pooling layer has got as input the location of the region proposal(xmin_roi,ymin_roi,h_roi,w_roi) and the previous feature map(14x14x512).


<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/id11.png"  height="400" width="700" ></center>
</div>

Now the location of ROI coordinates are in the units of the input image i.e. 224x224 pixels. But the layer on which we have to apply the ROI pooling operation is 14x14x512. As we are using VGG we will transform image (224 x 224 x 3) into (14 x 14 x 512) - height and width is divided by 16. we can map ROIs coordinates onto the feature map just by dividing them by 16.

<div style="color:black; background-color: #E9DAEE;">
In its depth, the convolutional feature map has encoded all the information for the image while maintaining the location of the "things" it has encoded relative to the original image. For example, if there was a red square on the top left of the image and the convolutional layers activate for it, then the information for that red square would still be on the top left of the convolutional feature map.
</div>
<br>
How the ROI pooling is done?

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/id12.gif"  height="400" width="700" ></center>
</div>

In the above image our region proposal is (0,3,5,7) and we divide that area into 4 regions since we want to have a ROI pooling layer of 2x2.

[How do you do ROI-Pooling on Areas smaller than the target size?](https://medium.com/r/?url=https%3A%2F%2Fstackoverflow.com%2Fquestions%2F48163961%2Fhow-do-you-do-roi-pooling-on-areas-smaller-than-the-target-size) if region proposal size is 5x5 and ROI pooling layer of size 7x7. If this happens, we resize to 35x35 just by copying 7 times each cell and then max-pooling back to 7x7.

After replacing the pooling layer, the authors also replaced the 1000 layer imagenet classification layer by a fully connected layer and softmax over K + 1 categories(+1 for Background) and category-specific bounding-box regressors.

### Training Fast-RCNN

What is the input to an Fast- RCNN?

Pretty much similar: So we have got an image, Region Proposals from the RPN strategy and the ground truths of the labels (labels, ground truth boxes)

Next we treat all region proposals with ≥ 0.5 IoU(Intersection over union) overlap with a ground-truth box as positive training example for that box's class and the rest as negative. This time we have a dense layer on top, and we use multi task loss.

So every ROI becomes a training example. The main difference is that there is concept of multi-task loss:

A Fast R-CNN network has two sibling output layers. The first outputs a discrete probability distribution (per RoI), p = (p0, . . . , pK), over K + 1 categories. As usual, p is computed by a softmax over the K+1 outputs of a fully connected layer. The second sibling layer outputs bounding-box regression offsets, t= (tx , ty , tw, th), for each of the K object classes. Each training RoI is labeled with a ground-truth class u and a ground-truth bounding-box regression target v. We use a multi-task loss L on each labeled RoI to jointly train for classification and bounding-box regression

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/id13.png"  height="400" width="700" ></center>
</div>

Where Lcls is the softmax classification loss and Lloc is the regression loss. u=0 is for BG class and hence we add to loss only when we have a boundary box for any of the other class. Further:

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/id14.png"  height="400" width="700" ></center>
</div>

### Problem:

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/id15.png"  height="400" width="700" ></center>
</div>

## Faster-RCNN

The next question that got asked was : Can the network itself do region proposals?

<div style="color:black; background-color: #E9DAEE;">
The intuition is that: With FastRCNN we're already computing an Activation Map in the CNN, why not run the Activation Map through a few more layers to find the interesting regions, and then finish off the forward pass by predicting the classes + bbox coordinates?
</div>


<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/id16.png"  height="400" width="700" ></center>
</div>

### How does the Region Proposal Network work?

One of the main idea in the paper is the idea of Anchors. Anchors are fixed bounding boxes that are placed throughout the image with different sizes and ratios that are going to be used for reference when first predicting object locations.

So first of all we define anchor centers on the image.

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/id17.png"  height="400" width="700" ></center>
</div>

The anchor centers are separated by 16 px in case of VGG16 network as the final convolution layer of (14x14x512) subsamples the image by a factor of 16(224/14).
This is how anchors look like:

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/id18.png"  height="400" width="700" ></center>
</div>

1. So we start with some predefined regions we think our objects could be with Anchors.
2. Our RPN Classifies which regions have the object and the offset of the object bounding box. 1 if IOU for anchor with bounding box>0.5 0 otherwise.
3. Non-Maximum suppression to reduce region proposals
4. Fast RCNN detection network on top of proposals

### Faster-RCNN Loss:

---

The whole network is then jointly trained with 4 losses:

1. RPN classify object / not object
2. RPN regress box coordinates offset
3. Final classification score (object classes)
4. Final box coordinates offset

## Results:


<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/id19.jpeg"  height="400" width="700" ></center>
</div>

**Disclaimer:** *This is my own understanding of these papers with inputs from many blogs and slides on the internet. Let me know if you find something wrong with my understanding. I will be sure to correct myself and post.*

## References:

1. [Transfer Learning](http://cs231n.github.io/transfer-learning/#tf)
2. [CS231 Object detection Lecture Slides](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture11.pdf)
3. [Efficient Graph-Based Image Segmentation](http://people.cs.uchicago.edu/~pff/papers/seg-ijcv.pdf)
4. [Rich feature hierarchies for accurate object detection and semantic segmentation(RCNN Paper)](https://arxiv.org/pdf/1311.2524.pdf)
5. [Selective Search for Object Recognition](https://medium.com/r/?url=http%3A%2F%2Fwww.huppelen.nl%2Fpublications%2FselectiveSearchDraft.pdf)
6. [ROI Pooling Explanation](https://deepsense.ai/region-of-interest-pooling-explained/)
7. [Faster RCNN Blog](https://towardsdatascience.com/fasterrcnn-explained-part-1-with-code-599c16568cff)
8. [StackOverflow](https://stackoverflow.com/questions/48163961/how-do-you-do-roi-pooling-on-areas-smaller-than-the-target-size)
9. [Faster RCNN Blog](https://medium.com/@smallfishbigsea/faster-r-cnn-explained-864d4fb7e3f8)
10. [Faster RCNN Blog](https://tryolabs.com/blog/2018/01/18/faster-r-cnn-down-the-rabbit-hole-of-modern-object-detection/)
11. [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://medium.com/r/?url=https%3A%2F%2Farxiv.org%2Fpdf%2F1506.01497.pdf)
12. [https://www.slideshare.net/WenjingChen7/deep-learning-for-object-detection](https://www.slideshare.net/WenjingChen7/deep-learning-for-object-detection)

