---
title:  End to End Pipeline for setting up Multiclass Image Classification for Data Scientists
date:  2020-06-24
draft: false
url : blog/2020/06/06/multiclass_image_classification_pytorch/
slug: multiclass_image_classification_pytorch
Category: Python

Keywords:
- Pandas
- Statistics

Categories:
- Deep Learning
- Computer Vision
- Awesome Guides

Tags:
- Deep Learning
- Artificial Intelligence
- Computer Vision
- Awesome Guides
- Image classification

description: In this post, we’ll create an end to end pipeline for image multiclass classification using Pytorch.This will include training the model, putting the model’s results in a form that can be shown to business partners, and functions to help deploy the model easily. As an added feature we will look at Test Time Augmentation using Pytorch also.

thumbnail : /images/multiclass_image_classification_pytorch/main.png
image : /images/multiclass_image_classification_pytorch/main.png
toc : false
type : post
---


*Have you ever wondered how Facebook takes care of the abusive and inappropriate images shared by some of its users? Or how Facebook’s tagging feature works? Or how Google Lens recognizes products through images?*

All of the above are examples of [image classification](https://lionbridge.ai/services/image-annotation/) in different settings. Multiclass image classification is a common task in computer vision, where we categorize an image into three or more classes.

In the past, I always used Keras for computer vision projects. However, recently when the opportunity to work on multiclass image classification presented itself, I decided to use PyTorch. I have already moved from Keras to PyTorch for all [NLP tasks](https://mlwhiz.com/blog/2019/01/06/pytorch_keras_conversion/), so why not vision, too?

> [PyTorch](https://coursera.pxf.io/jWG2Db) is powerful, and I also like its more pythonic structure.

***In this post, we’ll create an end to end pipeline for image multiclass classification using Pytorch.*** This will include training the model, putting the model’s results in a form that can be shown to business partners, and functions to help deploy the model easily. As an added feature we will look at Test Time Augmentation using Pytorch also.

But before we learn how to do image classification, let’s first look at transfer learning, the most common method for dealing with such problems.

---
## What is Transfer Learning?

Transfer learning is the process of repurposing knowledge from one task to another. From a modelling perspective, this means using a model trained on one dataset and fine-tuning it for use with another. But why does it work?

Let’s start with some background. Every year the visual recognition community comes together for a very particular challenge: [The Imagenet Challenge](http://image-net.org/explore). The task in this challenge is to classify 1,000,000 images into 1,000 categories.

This challenge has already resulted in researchers training big convolutional deep learning models. The results have included great models like Resnet50 and Inception.

But, what does it mean to train a neural model? Essentially, it means the researchers have learned the weights for a neural network after training the model on a million images.

So, what if we could get those weights? We could then use them and load them into our own neural networks model to predict on the test dataset, right? Actually, we can go even further than that; we can add an extra layer on top of the neural network these researchers have prepared to classify our own dataset.

> While the exact workings of these complex models is still a mystery, we do know that the lower convolutional layers capture low-level image features like edges and gradients. In comparison, higher convolutional layers capture more and more intricate details, such as body parts, faces, and other compositional features.

![Source: [Visualizing and Understanding Convolutional Networks](https://arxiv.org/pdf/1311.2901.pdf). You can see how the first few layers capture basic shapes, and the shapes become more and more complex in the later layers.](/images/multiclass_image_classification_pytorch/0.png)*Source: [Visualizing and Understanding Convolutional Networks](https://arxiv.org/pdf/1311.2901.pdf). You can see how the first few layers capture basic shapes, and the shapes become more and more complex in the later layers.*

In the example above from ZFNet (a variant of Alexnet), one of the first convolutional neural networks to achieve success on the Imagenet task, you can see how the lower layers capture lines and edges, and the later layers capture more complex features. The final fully-connected layers are generally assumed to capture information that is relevant for solving the respective task, e.g. ZFNet’s fully-connected layers indicate which features are relevant for classifying an image into one of 1,000 object categories.

For a new vision task, it is possible for us to simply use the off-the-shelf features of a state-of-the-art CNN pre-trained on ImageNet, and train a new model on these extracted features.

The intuition behind this idea is that a model trained to recognize animals might also be used to recognize cats vs dogs. In our case,
> a model that has been trained on 1000 different categories has seen a lot of real-world information, and we can use this information to create our own custom classifier.

***So that’s the theory and intuition. How do we get it to actually work? Let’s look at some code. You can find the complete code for this post on [Github](https://github.com/MLWhiz/data_science_blogs/tree/master/compvisblog).***

---
## Data Exploration

We will start with the [Boat Dataset](https://www.kaggle.com/clorichel/boat-types-recognition/version/1) from Kaggle to understand the multiclass image classification problem. This dataset contains about 1,500 pictures of boats of different types: buoys, cruise ships, ferry boats, freight boats, gondolas, inflatable boats, kayaks, paper boats, and sailboats. Our goal is to create a model that looks at a boat image and classifies it into the correct category.

Here’s a sample of images from the dataset:

![](/images/multiclass_image_classification_pytorch/1.png)

And here are the category counts:

![](/images/multiclass_image_classification_pytorch/2.png)

Since the categories *“freight boats”, “inflatable boats”
, and “boats”* don’t have a lot of images; we will be removing these categories when we train our model.

---
## Creating the required Directory Structure

Before we can go through with training our deep learning models, we need to create the required directory structure for our images. Right now, our data directory structure looks like:

    images
        sailboat
        kayak
        .
        .

We need our images to be contained in 3 folders train, val and test. We will then train on the images in train dataset, validate on the ones in the val dataset and finally test them on images in the test dataset.

    data
        train
            sailboat
            kayak
            .
            .
        val
            sailboat
            kayak
            .
            .
        test
            sailboat
            kayak
            .
            .

You might have your data in a different format, but I have found that apart from the usual libraries, the glob.glob and os.system functions are very helpful. Here you can find the complete [data preparation code](https://github.com/MLWhiz/data_science_blogs/blob/master/compvisblog/Boats_DataExploration.ipynb). Now let’s take a quick look at some of the not-so-used libraries that I found useful while doing data prep.

### What is glob.glob?

Simply, glob lets you get names of files or folders in a directory using a regex. For example, you can do something like:

    from glob import glob
    categories = glob(“images/*”)
    print(categories)
    ------------------------------------------------------------------
    ['images/kayak', 'images/boats', 'images/gondola', 'images/sailboat', 'images/inflatable boat', 'images/paper boat', 'images/buoy', 'images/cruise ship', 'images/freight boat', 'images/ferry boat']

### What is os.system?

os.system is a function in os library which lets you run any command-line function in python itself. I generally use it to run Linux functions, but it can also be used to run R scripts within python as shown [here](https://towardsdatascience.com/python-pro-tip-want-to-use-r-java-c-or-any-language-in-python-d304be3a0559). For example, I use it in my data preparation to copy files from one directory to another after getting the information from a pandas data frame. I also use [f string formatting](https://towardsdatascience.com/how-and-why-to-use-f-strings-in-python3-adbba724b251).

    import os

    for i,row in fulldf.iterrows():
        # Boat category
        cat = row['category']
        # section is train,val or test
        section = row['type']
        # input filepath to copy
        ipath = row['filepath']
        # output filepath to paste
        opath = ipath.replace(f"images/",f"data/{section}/")
        # running the cp command
        os.system(f"cp '{ipath}' '{opath}'")

Now since we have our data in the required folder structure, we can move on to more exciting parts.

---
## Data Preprocessing

### Transforms:

**1. Imagenet Preprocessing**

In order to use our images with a network trained on the Imagenet dataset, we need to preprocess our images in the same way as the Imagenet network. For that, we need to rescale the images to 224×224 and normalize them as per Imagenet standards. We can use the torchvision transforms library to do that. Here we take a CenterCrop of 224×224 and normalize as per Imagenet standards. The operations defined below happen sequentially. You can find a list of all transforms provided by [PyTorch here](https://pytorch.org/docs/stable/torchvision/transforms.html).

    transforms.Compose([
            transforms.CenterCrop(size=224),  
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])  
        ])

**2. Data Augmentations**

We can do a lot more preprocessing for data augmentations. Neural networks work better with a lot of data. [Data augmentation](https://lionbridge.ai/articles/data-augmentation-with-machine-learning-an-overview/) is a strategy which we use at training time to increase the amount of data we have.

For example, we can flip the image of a boat horizontally, and it will still be a boat. Or we can randomly crop images or add color jitters. Here is the image transforms dictionary I have used that applies to both the Imagenet preprocessing as well as augmentations. This dictionary contains the various transforms we have for the train, test and validation data as used in this [great post](https://towardsdatascience.com/transfer-learning-with-convolutional-neural-networks-in-pytorch-dd09190245ce). As you’d expect, we don’t apply the horizontal flips or other data augmentation transforms to the test data and validation data because we don’t want to get predictions on an augmented image.

    # Image transformations
    image_transforms = {
        # Train uses data augmentation
        'train':
        transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),  # Image net standards
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])  # Imagenet standards
        ]),
        # Validation does not use augmentation
        'valid':
        transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),

            # Test does not use augmentation
        'test':
        transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

Here is an example of the train transforms applied to an image in the training dataset. Not only do we get a lot of different images from a single image, but it also helps our network become invariant to the object orientation.

    ex_img = Image.open('/home/rahul/projects/compvisblog/data/train/cruise ship/cruise-ship-oasis-of-the-seas-boat-water-482183.jpg')

    t = image_transforms['train']
    plt.figure(figsize=(24, 24))

    for i in range(16):
        ax = plt.subplot(4, 4, i + 1)
        _ = imshow_tensor(t(ex_img), ax=ax)

    plt.tight_layout()

![](/images/multiclass_image_classification_pytorch/3.png)

### DataLoaders

The next step is to provide the training, validation, and test dataset locations to PyTorch. We can do this by using the PyTorch datasets and DataLoader class. This part of the code will mostly remain the same if we have our data in the required directory structures.

    # Datasets from folders

    traindir = "data/train"
    validdir = "data/val"
    testdir = "data/test"

    data = {
        'train':
        datasets.ImageFolder(root=traindir, transform=image_transforms['train']),
        'valid':
        datasets.ImageFolder(root=validdir, transform=image_transforms['valid']),
        'test':
        datasets.ImageFolder(root=testdir, transform=image_transforms['test'])
    }

    # Dataloader iterators, make sure to shuffle
    dataloaders = {
        'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True,num_workers=10),
        'val': DataLoader(data['valid'], batch_size=batch_size, shuffle=True,num_workers=10),
        'test': DataLoader(data['test'], batch_size=batch_size, shuffle=True,num_workers=10)
    }

These dataloaders help us to iterate through the dataset. For example, we will use the dataloader below in our model training. The data variable will contain data in the form (batch_size, color_channels, height, width) while the target is of shape (batch_size) and hold the label information.

    train_loader = dataloaders['train']
    for ii, (data, target) in enumerate(train_loader):

---
## Modeling

### 1. Create the model using a pre-trained model

Right now these following pre-trained models are available to use in the torchvision library:

* [AlexNet](https://arxiv.org/abs/1404.5997)

* [VGG](https://arxiv.org/abs/1409.1556)

* [ResNet](https://arxiv.org/abs/1512.03385)

* [SqueezeNet](https://arxiv.org/abs/1602.07360)

* [DenseNet](https://arxiv.org/abs/1608.06993)

* [Inception](https://arxiv.org/abs/1512.00567) v3

* [GoogLeNet](https://arxiv.org/abs/1409.4842)

* [ShuffleNet](https://arxiv.org/abs/1807.11164) v2

* [MobileNet](https://arxiv.org/abs/1801.04381) v2

* [ResNeXt](https://arxiv.org/abs/1611.05431)

* [Wide ResNet](https://pytorch.org/docs/stable/torchvision/models.html#wide-resnet)

* [MNASNet](https://arxiv.org/abs/1807.11626)

Here I will be using resnet50 on our dataset, but you can effectively use any other model too as per your choice.

    from torchvision import models
    model = models.resnet50(pretrained=True)

We start by freezing our model weights since we don’t want to change the weights for the renet50 models.

    # Freeze model weights
    for param in model.parameters():
        param.requires_grad = False

The next thing we need to do is to replace the linear classification layer in the model by our custom classifier. I have found that to do this, it is better first to see the model structure to determine what is the final linear layer. We can do this simply by printing the model object:

    print(model)
    ------------------------------------------------------------------
    ResNet(
      (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
       .
       .
       .
       .

    (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
      )  
    (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
      **(fc): Linear(in_features=2048, out_features=1000, bias=True)**
    )

Here we find that the final linear layer that takes the input from the convolutional layers is named fc

We can now simply replace the fc layer using our custom neural network. This neural network takes input from the previous layer to fc and gives the log softmax output of shape (batch_size x n_classes).

    n_inputs = model.fc.in_features
    model.fc = nn.Sequential(
                          nn.Linear(n_inputs, 256),
                          nn.ReLU(),
                          nn.Dropout(0.4),
                          nn.Linear(256, n_classes),                   
                          nn.LogSoftmax(dim=1))

Please note that the new layers added now are fully trainable by default.

### 2. Load the model on GPU

We can use a single GPU or multiple GPU(if we have them) using DataParallel from PyTorch. Here is what we can use to detect the GPU as well as the number of GPUs to load the model on GPU. Right now I am training my models on dual NVIDIA Titan RTX GPUs.

    # Whether to train on a gpu
    train_on_gpu = cuda.is_available()
    print(f'Train on gpu: {train_on_gpu}')

    # Number of gpus
    if train_on_gpu:
        gpu_count = cuda.device_count()
        print(f'{gpu_count} gpus detected.')
        if gpu_count > 1:
            multi_gpu = True
        else:
            multi_gpu = False

    if train_on_gpu:
        model = model.to('cuda')

    if multi_gpu:
        model = nn.DataParallel(model)

### 3. Define criterion and optimizers

One of the most important things to notice when you are training any model is the choice of loss-function and the optimizer used. Here we want to use categorical cross-entropy as we have got a multiclass classification problem and the [Adam](https://cs231n.github.io/neural-networks-3/#ada) optimizer, which is the most commonly used optimizer. But since we are applying a LogSoftmax operation on the output of our model, we will be using the NLL loss.

    from torch import optim

    criteration = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())

### 4. Training the model

Given below is the full code used to train the model. It might look pretty big on its own, but essentially what we are doing is as follows:

* Start running epochs. In each epoch-

* Set the model mode to train using model.train().

* Loop through the data using the train dataloader.

* Load your data to the GPU using the data, target = data.cuda(), target.cuda() command

* Set the existing gradients in the optimizer to zero using optimizer.zero_grad()

* Run the forward pass through the batch using output = model(data)

* Compute loss using loss = criterion(output, target)

* Backpropagate the losses through the network using loss.backward()

* Take an optimizer step to change the weights in the whole network using optimizer.step()

* All the other steps in the training loop are just to maintain the history and calculate accuracy.

* Set the model mode to eval using model.eval().

* Get predictions for the validation data using valid_loader and calculate valid_loss and valid_acc

* Print the validation loss and validation accuracy results every print_every epoch.

* Save the best model based on validation loss.

* **Early Stopping:** If the cross-validation loss doesn’t improve for max_epochs_stop stop the training and load the best available model with the minimum validation loss.

```py
def train(model,
          criterion,
          optimizer,
          train_loader,
          valid_loader,
          save_file_name,
          max_epochs_stop=3,
          n_epochs=20,
          print_every=1):
    """Train a PyTorch Model
    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats
    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """

    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    valid_max_acc = 0
    history = []

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')

    overall_start = timer()

    # Main loop
    for epoch in range(n_epochs):

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0

        # Set to training
        model.train()
        start = timer()

        # Training loop
        for ii, (data, target) in enumerate(train_loader):
            # Tensors to gpu
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs are log probabilities
            output = model(data)

            # Loss and backpropagation of gradients
            loss = criterion(output, target)
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)

            # Calculate accuracy by finding max log probability
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * data.size(0)

            # Track training progress
            print(
                f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                end='\r')

        # After training loops ends, start validation
        else:
            model.epochs += 1

            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()

                # Validation loop
                for data, target in valid_loader:
                    # Tensors to gpu
                    if train_on_gpu:
                        data, target = data.cuda(), target.cuda()

                    # Forward pass
                    output = model(data)

                    # Validation loss
                    loss = criterion(output, target)
                    # Multiply average loss times the number of examples in batch
                    valid_loss += loss.item() * data.size(0)

                    # Calculate validation accuracy
                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(
                        correct_tensor.type(torch.FloatTensor))
                    # Multiply average accuracy times the number of examples
                    valid_acc += accuracy.item() * data.size(0)

                # Calculate average losses
                train_loss = train_loss / len(train_loader.dataset)
                valid_loss = valid_loss / len(valid_loader.dataset)

                # Calculate average accuracy
                train_acc = train_acc / len(train_loader.dataset)
                valid_acc = valid_acc / len(valid_loader.dataset)

                history.append([train_loss, valid_loss, train_acc, valid_acc])

                # Print training and validation results
                if (epoch + 1) % print_every == 0:
                    print(
                        f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
                    )
                    print(
                        f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
                    )

                # Save the model if validation loss decreases
                if valid_loss < valid_loss_min:
                    # Save model
                    torch.save(model.state_dict(), save_file_name)
                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    valid_best_acc = valid_acc
                    best_epoch = epoch

                # Otherwise increment count of epochs with no improvement
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= max_epochs_stop:
                        print(
                            f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
                        )
                        total_time = timer() - overall_start
                        print(
                            f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                        )

                        # Load the best state dict
                        model.load_state_dict(torch.load(save_file_name))
                        # Attach the optimizer
                        model.optimizer = optimizer

                        # Format history
                        history = pd.DataFrame(
                            history,
                            columns=[
                                'train_loss', 'valid_loss', 'train_acc',
                                'valid_acc'
                            ])
                        return model, history

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
    )
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'
    )
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
    return model, history

# Running the model
model, history = train(
    model,
    criterion,
    optimizer,
    dataloaders['train'],
    dataloaders['val'],
    save_file_name=save_file_name,
    max_epochs_stop=3,
    n_epochs=100,
    print_every=1)

```
Here is the output from running the above code. Just showing the last few epochs. The validation accuracy started at ~55% in the first epoch, and we ended up with a validation accuracy of ~90%.

![](/images/multiclass_image_classification_pytorch/4.png)

And here are the training curves showing the loss and accuracy metrics:

![](/images/multiclass_image_classification_pytorch/5.png)

![Training curves](/images/multiclass_image_classification_pytorch/6.png)

---
## Inference and Model Results

We want our results in different ways to use our model. For one, we require test accuracies and confusion matrices. All of the code for creating these results is in the code notebook.

### 1. Test Results

The overall accuracy of the test model is:

    Overall Accuracy: 88.65 %

Here is the confusion matrix for results on the test dataset.

![](/images/multiclass_image_classification_pytorch/7.png)

We can also look at the category wise accuracies. I have also added the train counts to see the results from a new perspective.

![](/images/multiclass_image_classification_pytorch/8.png)

### 2. Visualizing Predictions for Single Image

For deployment purposes, it helps to be able to get predictions for a single image. You can get the code from the notebook.

![](/images/multiclass_image_classification_pytorch/9.png)

### 3. Visualizing Predictions for a Category

We can also see the category wise results for debugging purposes and presentations.

![](/images/multiclass_image_classification_pytorch/10.png)

### 4. Test results with Test Time Augmentation

We can also do test time augmentation to increase our test accuracy. Here I am using a new test data loader and transforms:

    # Image transformations
    tta_random_image_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),  # Image net standards
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])  # Imagenet standards
        ])

    # Datasets from folders
    ttadata = {
        'test':
        datasets.ImageFolder(root=testdir, transform=tta_random_image_transforms)
    }

    # Dataloader iterators
    ttadataloader = {
        'test': DataLoader(ttadata['test'], batch_size=512, shuffle=False,num_workers=10)
    }

We can then get the predictions on the test set using the below function:

```py
def tta_preds_n_averaged(model, test_loader,n=5):
    """Returns the TTA preds from a trained PyTorch model
    Params
    --------
        model (PyTorch model): trained cnn for inference
        test_loader (PyTorch DataLoader): test dataloader

    Returns
    --------
        results (array): results for each category
    """
    # Hold results
    results = np.zeros((len(test_loader.dataset), n_classes))
    bs = test_loader.batch_size
    model.eval()
    with torch.no_grad():
        #aug loop:
        for _ in range(n):
            # Testing loop
            tmp_pred = np.zeros((len(test_loader.dataset), n_classes))
            for i,(data, targets) in enumerate(tqdm.tqdm(test_loader)):

                # Tensors to gpu
                if train_on_gpu:
                    data, targets = data.to('cuda'), targets.to('cuda')

                # Raw model output
                out = model(data)
                tmp_pred[i*bs:(i+1)*bs] = np.array(out.cpu())

            results+=tmp_pred
    return results/n
```
In the function above, I am applying the tta_random_image_transforms to each image 5 times before getting its prediction. The final prediction is the average of all five predictions. When we use TTA over the whole test dataset, we noticed that the accuracy increased by around 1%

    TTA Accuracy: 89.71%

Also, here is the results for TTA compared to normal results category wise:

![](/images/multiclass_image_classification_pytorch/11.png)

In this small dataset, the TTA might not seem to add much value, but I have noticed that it adds value with big datasets.

---
## Conclusion

In this post, I talked about the end to end pipeline for working on a multiclass image classification project using PyTorch. We worked on creating some readymade code to train a model using transfer learning, visualize the results, use Test time augmentation, and got predictions for a single image so that we can deploy our model when needed using any tool like [Streamlit](https://towardsdatascience.com/how-to-write-web-apps-using-simple-python-for-data-scientists-a227a1a01582).

You can find the complete code for this post on [Github](https://github.com/MLWhiz/data_science_blogs/tree/master/compvisblog).

If you would like to learn more about Image Classification and Convolutional Neural Networks take a look at the [Deep Learning Specialization](https://coursera.pxf.io/7mKnnY) from Andrew Ng. Also, to learn more about PyTorch and start from the basics, you can take a look at the [Deep Neural Networks with PyTorch](https://coursera.pxf.io/jWG2Db) course offered by IBM.

Thanks for the read. I am going to be writing more beginner-friendly posts in the future too. Follow me up at [Medium](https://mlwhiz.medium.com/) or Subscribe to my [blog](https://mlwhiz.ck.page/a9b8bda70c).

Also, a small disclaimer — There might be some affiliate links in this post to relevant resources, as sharing knowledge is never a bad idea.

*This post was first published [here](https://lionbridge.ai/articles/end-to-end-multiclass-image-classification-using-pytorch-and-transfer-learning/).*
