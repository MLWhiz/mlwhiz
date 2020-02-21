---
title: "A Layman guide to moving from Keras to Pytorch"
date:   2019-01-06
draft: false
url : blog/2019/01/06/pytorch_keras_conversion/
slug: pytorch_keras_conversion
Category: deep learning, nlp, kaggle
Keywords: 
- artificial intelligence
-  deep learning
- kaggle
-  pytorch
-  keras to pytorch
-  keras vs pytorch
-  keras is better than pytorch
-  pytorch better than keras
-  pytorch for text classification
-  kaggle text classification
-  deep learning and nlp with pytorch 
- text classification pytorch
-  birnn
-  bidirectional RNN
-  bidirectional LSTM for text
- bidirectional GRU for text
-  Attention models for text
Tags: 
- deep learning
- NLP
- Python
description: Recently I started up with a competition on kaggle on text classification, and as a part of the competition I had to somehow move to Pytorch to get deterministic results. Here are some of my findings.
toc : false
thumbnail: images/artificial-neural-network.png
images:
 - https://mlwhiz.com/images/artificial-neural-network.png
---

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/artificial-neural-network.png"  height="350" width="700" ></center>
</div>

Recently I started up with a competition on kaggle on text classification, and as a part of the competition, I had to somehow move to Pytorch to get deterministic results. Now I have always worked with Keras in the past and it has given me pretty good results, but somehow I got to know that the **CuDNNGRU/CuDNNLSTM layers in keras are not deterministic**, even after setting the seeds. So Pytorch did come to rescue. And am  I  glad that I moved. 

As a **side note**: if you want to know more about **NLP**, I would like to recommend this awesome course on **[Natural Language Processing](https://www.coursera.org/specializations/aml?siteID=lVarvwc5BD0-AqkGMb7JzoCMW0Np1uLfCA&utm_content=2&utm_medium=partners&utm_source=linkshare&utm_campaign=lVarvwc5BD0)** in the **[Advanced machine learning specialization](https://www.coursera.org/specializations/aml?siteID=lVarvwc5BD0-AqkGMb7JzoCMW0Np1uLfCA&utm_content=2&utm_medium=partners&utm_source=linkshare&utm_campaign=lVarvwc5BD0)**. You can start for free with the 7-day Free Trial. This course covers a wide range of tasks in Natural Language Processing from basic to advanced: Sentiment Analysis, summarization, dialogue state tracking, to name a few. 

Also take a look at my other post: [Text Preprocessing Methods for Deep Learning](/blog/2019/01/17/deeplearning_nlp_preprocess/), which talks about different preprocessing techniques you can use for your NLP task and [What Kagglers are using for Text Classification](/blog/2018/12/17/text_classification/), which talks about various deep learning models in use in NLP.

Ok back to the task at hand. *While Keras is great to start with deep learning, with time you are going to resent some of its limitations.* I sort of thought about moving to Tensorflow. It seemed like a good transition as TF is the backend of Keras. But was it hard? With the whole `session.run` commands and tensorflow sessions, I was sort of confused. It was not Pythonic at all.

Pytorch helps in that since it seems like the **python way to do things**. You have things under your control and you are not losing anything on the performance front. In the words of Andrej Karpathy:

<center><blockquote class="twitter-tweet" data-lang="en"><p lang="en" dir="ltr">I&#39;ve been using PyTorch a few months now and I&#39;ve never felt better. I have more energy. My skin is clearer. My eye sight has improved.</p>&mdash; Andrej Karpathy (@karpathy) <a href="https://twitter.com/karpathy/status/868178954032513024?ref_src=twsrc%5Etfw">May 26, 2017</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script></center>

So without further ado let me translate Keras to Pytorch for you. 

## The Classy way to write your network?

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/structured.jpeg"  height="400" width="700" ></center>
</div>

Ok, let us create an example network in keras first which we will try to port into Pytorch. Here I would like to give a piece of advice too. When you try to move from Keras to Pytorch **take any network you have and try porting it to Pytorch**. It will make you understand Pytorch in a much better way. Here I am trying to write one of the networks that gave pretty good results in the Quora Insincere questions classification challenge for me. This model has all the bells and whistles which at least any Text Classification deep learning network could contain with its GRU, LSTM and embedding layers and also a meta input layer. And thus would serve as a good example. Also if you want to read up more on how the BiLSTM/GRU and Attention model work do visit my post [here](/blog/2018/12/17/text_classification/).

```py
def get_model(features,clipvalue=1.,num_filters=40,dropout=0.1,embed_size=501):
    features_input = Input(shape=(features.shape[1],))
    inp = Input(shape=(maxlen, ))
    
    # Layer 1: Word2Vec Embeddings.
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    
    # Layer 2: SpatialDropout1D(0.1)
    x = SpatialDropout1D(dropout)(x)
    
    # Layer 3: Bidirectional CuDNNLSTM
    x = Bidirectional(LSTM(num_filters, return_sequences=True))(x)

    # Layer 4: Bidirectional CuDNNGRU
    x, x_h, x_c = Bidirectional(GRU(num_filters, return_sequences=True, return_state = True))(x)  
    
    # Layer 5: some pooling operations
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    
    # Layer 6: A concatenation of the last state, maximum pool, average pool and 
    # additional features
    x = concatenate([avg_pool, x_h, max_pool,features_input])
    
    # Layer 7: A dense layer
    x = Dense(16, activation="relu")(x)

    # Layer 8: A dropout layer
    x = Dropout(0.1)(x)
    
    # Layer 9: Output dense layer with one output for our Binary Classification problem.
    outp = Dense(1, activation="sigmoid")(x)

    # Some keras model creation and compiling
    model = Model(inputs=[inp,features_input], outputs=outp)
    adam = optimizers.adam(clipvalue=clipvalue)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model
```

So a model in pytorch is defined as a class(therefore a little more classy) which inherits from `nn.module` . Every class necessarily contains an `__init__` procedure block and a block for the `forward` pass.

- In the `__init__` part the user defines all the layers the network is going to have but doesn't yet define how those layers would be connected to each other

- In the forward pass block, the user defines how data flows from one layer to another inside the network. 

#### Why is this Classy?

Obviously classy because of Classes. Duh! But jokes apart, I found it beneficial due to a couple of reasons:

1) It gives you a **lot of control** on how your network is built. 

2) You understand a lot about the network when you are building it since you have to specify input and output dimensions. So ** fewer chances of error**. (Although this one is really up to the skill level)

3) **Easy to debug** networks. Any time you find any problem with the network just use something like `print("avg_pool", avg_pool.size())` in the forward pass to check the sizes of the layer and you will debug the network easily

4) You can **return multiple outputs** from the forward layer. This is pretty helpful in the Encoder-Decoder architecture where you can return both the encoder and decoder output. Or in the case of autoencoder where you can return the output of the model and the hidden layer embedding for the data.

5) **Pytorch tensors work in a very similar manner to numpy arrays**. For example, I could have used Pytorch Maxpool function to write the maxpool layer but `max_pool, _ = torch.max(h_gru, 1)` will also work.

6) You can set up **different layers with different initialization schemes**. Something you won't be able to do in Keras. For example, in the below network I have changed the initialization scheme of my LSTM layer. The LSTM layer has different initializations for biases, input layer weights, and hidden layer weights. 

7) Wait until you see the **training loop in Pytorch** You will be amazed at the sort of **control** it provides. 

Now the same model in Pytorch will look like something like this. Do go through the code comments to understand more on how to port. 

```py
class Alex_NeuralNet_Meta(nn.Module):
    def __init__(self,hidden_size,lin_size, embedding_matrix=embedding_matrix):
        super(Alex_NeuralNet_Meta, self).__init__()

        # Initialize some parameters for your model
        self.hidden_size = hidden_size
        drp = 0.1

        # Layer 1: Word2Vec Embeddings.
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        # Layer 2: Dropout1D(0.1)
        self.embedding_dropout = nn.Dropout2d(0.1)

        # Layer 3: Bidirectional CuDNNLSTM
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)

        # Layer 4: Bidirectional CuDNNGRU
        self.gru = nn.GRU(hidden_size*2, hidden_size, bidirectional=True, batch_first=True)

        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)

        # Layer 7: A dense layer
        self.linear = nn.Linear(hidden_size*6 + features.shape[1], lin_size)
        self.relu = nn.ReLU()
        
        # Layer 8: A dropout layer 
        self.dropout = nn.Dropout(drp)

        # Layer 9: Output dense layer with one output for our Binary Classification problem.
        self.out = nn.Linear(lin_size, 1)

    def forward(self, x):
        '''
        here x[0] represents the first element of the input that is going to be passed. 
        We are going to pass a tuple where first one contains the sequences(x[0])
        and the second one is a additional feature vector(x[1])
        '''
        h_embedding = self.embedding(x[0])
        # Based on comment by Ivank to integrate spatial dropout. 
        embeddings = h_embedding.unsqueeze(2)    # (N, T, 1, K)
        embeddings = embeddings.permute(0, 3, 2, 1)  # (N, K, 1, T)
        embeddings = self.embedding_dropout(embeddings)  # (N, K, 1, T), some features are masked
        embeddings = embeddings.permute(0, 3, 2, 1)  # (N, T, 1, K)
        h_embedding = embeddings.squeeze(2)  # (N, T, K)
        #h_embedding = torch.squeeze(self.embedding_dropout(torch.unsqueeze(h_embedding, 0)))
        
        #print("emb", h_embedding.size())
        h_lstm, _ = self.lstm(h_embedding)
        #print("lst",h_lstm.size())
        h_gru, hh_gru = self.gru(h_lstm)
        hh_gru = hh_gru.view(-1, 2*self.hidden_size )
        #print("gru", h_gru.size())
        #print("h_gru", hh_gru.size())

        # Layer 5: is defined dynamically as an operation on tensors.
        avg_pool = torch.mean(h_gru, 1)
        max_pool, _ = torch.max(h_gru, 1)
        #print("avg_pool", avg_pool.size())
        #print("max_pool", max_pool.size())
        
        # the extra features you want to give to the model
        f = torch.tensor(x[1], dtype=torch.float).cuda()
        #print("f", f.size())

        # Layer 6: A concatenation of the last state, maximum pool, average pool and 
        # additional features
        conc = torch.cat(( hh_gru, avg_pool, max_pool,f), 1)
        #print("conc", conc.size())

        # passing conc through linear and relu ops
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)
        # return the final output
        return out
```


Hope you are still there with me. One thing I would like to emphasize here is that you need to code something up in Pytorch to really understand how it works. And know that once you do that you would be glad that you put in the effort. On to the next section.

## Tailored or Readymade: The Best Fit with a highly customizable Training Loop

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/sewing-machine.jpg"  height="300" width="700" ></center>
</div>

In the above section I wrote that you will be amazed once you saw the training loop. That was an exaggeration. On the first try you will be a little baffled/confused. But as soon as you read through the loop more than once it will make a lot of intuituve sense. Once again read up the comments and the code to gain a better understanding. 

This training loop does k-fold cross-validation on your training data and outputs Out-of-fold train_preds and test_preds averaged over the runs on the test data. I apologize if the flow looks something straight out of a kaggle competition, but if you understand this you would be able to create a training loop for your own workflow. And that is the beauty of Pytorch. 

So a brief summary of this loop are as follows:

- Create stratified splits using train data
- Loop through the splits.
    - Convert your train and CV data to tensor and load your data to the GPU using the 
`X_train_fold = torch.tensor(x_train[train_idx.astype(int)], dtype=torch.long).cuda()` command
    - Load the model onto the GPU using the `model.cuda()` command
    - Define Loss function, Scheduler and Optimizer
    - create `train_loader`    and     valid_loader` to iterate through batches.
    - Start running epochs. In each epoch
        - Set the model mode to train using `model.train()`. 
        - Go through the batches in `train_loader` and run the forward pass
        - Run a scheduler step to change the learning rate
        - Compute loss
        - Set the existing gradients in the optimizer to zero
        - Backpropagate the losses through the network
        - Clip the gradients
        - Take an optimizer step to change the weights in the whole network
        - Set the model mode to eval using `model.eval()`. 
        - Get predictions for the validation data using `valid_loader` and store in variable `valid_preds_fold`
        - Calculate Loss and print
    - After all epochs are done. Predict the test data and store the predictions. These predictions will be averaged at the end of the split loop to get the final `test_preds`
    - Get Out-of-fold(OOF) predictions for train set using `train_preds[valid_idx] = valid_preds_fold`
    - These OOF predictions can then be used to calculate the Local CV score for your model.


```py
def pytorch_model_run_cv(x_train,y_train,features,x_test, model_obj, feats = False,clip = True):
    seed_everything()
    avg_losses_f = []
    avg_val_losses_f = []
    # matrix for the out-of-fold predictions
    train_preds = np.zeros((len(x_train)))
    # matrix for the predictions on the test set
    test_preds = np.zeros((len(x_test)))
    splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(x_train, y_train))
    for i, (train_idx, valid_idx) in enumerate(splits):
        seed_everything(i*1000+i)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        if feats:
            features = np.array(features)
        x_train_fold = torch.tensor(x_train[train_idx.astype(int)], dtype=torch.long).cuda()
        y_train_fold = torch.tensor(y_train[train_idx.astype(int), np.newaxis], dtype=torch.float32).cuda()
        if feats:
            kfold_X_features = features[train_idx.astype(int)]
            kfold_X_valid_features = features[valid_idx.astype(int)]
        x_val_fold = torch.tensor(x_train[valid_idx.astype(int)], dtype=torch.long).cuda()
        y_val_fold = torch.tensor(y_train[valid_idx.astype(int), np.newaxis], dtype=torch.float32).cuda()
        
        model = copy.deepcopy(model_obj)

        model.cuda()

        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')

        step_size = 300
        base_lr, max_lr = 0.001, 0.003   
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                 lr=max_lr)
        
        ################################################################################################
        scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr,
                   step_size=step_size, mode='exp_range',
                   gamma=0.99994)
        ###############################################################################################

        train = MyDataset(torch.utils.data.TensorDataset(x_train_fold, y_train_fold))
        valid = MyDataset(torch.utils.data.TensorDataset(x_val_fold, y_val_fold))
        
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

        print(f'Fold {i + 1}')
        for epoch in range(n_epochs):
            start_time = time.time()
            model.train()

            avg_loss = 0.  
            for i, (x_batch, y_batch, index) in enumerate(train_loader):
                if feats:       
                    f = kfold_X_features[index]
                    y_pred = model([x_batch,f])
                else:
                    y_pred = model(x_batch)

                if scheduler:
                    scheduler.batch_step()

                # Compute and print loss.
                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                if clip:
                    nn.utils.clip_grad_norm_(model.parameters(),1)
                optimizer.step()
                avg_loss += loss.item() / len(train_loader)
                
            model.eval()
            
            valid_preds_fold = np.zeros((x_val_fold.size(0)))
            test_preds_fold = np.zeros((len(x_test)))
            
            avg_val_loss = 0.
            for i, (x_batch, y_batch,index) in enumerate(valid_loader):
                if feats:
                    f = kfold_X_valid_features[index]            
                    y_pred = model([x_batch,f]).detach()
                else:
                    y_pred = model(x_batch).detach()
                
                avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
                valid_preds_fold[index] = sigmoid(y_pred.cpu().numpy())[:, 0]
            
            elapsed_time = time.time() - start_time 
            print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
                epoch + 1, n_epochs, avg_loss, avg_val_loss, elapsed_time))
        avg_losses_f.append(avg_loss)
        avg_val_losses_f.append(avg_val_loss) 
        # predict all samples in the test set batch per batch
        for i, (x_batch,) in enumerate(test_loader):
            if feats:
                f = test_features[i * batch_size:(i+1) * batch_size]
                y_pred = model([x_batch,f]).detach()
            else:
                y_pred = model(x_batch).detach()

            test_preds_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]
            
        train_preds[valid_idx] = valid_preds_fold
        test_preds += test_preds_fold / len(splits)

    print('All \t loss={:.4f} \t val_loss={:.4f} \t '.format(np.average(avg_losses_f),np.average(avg_val_losses_f)))
    return train_preds, test_preds

```

#### But Why? Why so much code?

Okay. I get it. That was probably a handful. What you could have done with a simple`.fit` in keras, takes a lot of code to accomplish in Pytorch. But understand that you get a lot of power too. Some use cases for you to understand:

- While in Keras you have prespecified schedulers like `ReduceLROnPlateau` (and it is a task to write them), in Pytorch you can experiment like crazy. **If you know how to write Python you are going to get along just fine**
-  Want to change the structure of your model between the epochs. Yeah you can do it. Changing the input size for convolution networks on the fly.
- And much more. It is only your imagination that will stop you.

## Wanna Run it Yourself?


<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/tools.jpg" alt="You have all the tools it seems" height="400" width="700" ></center>
</div>

So another small confession here. The code above will not run as is as there are some code artifacts which I have not shown here. I did this in favor of making the post more readable. Like you see the `seed_everything`, `MyDataset` and `CyclicLR` (From Jeremy Howard Course) functions and classes in the code above which are not really included with Pytorch. But fret not my friend. I have tried to write a [Kaggle Kernel](https://www.kaggle.com/mlwhiz/third-place-model-for-toxic-comments-in-pytorch/edit) with the whole running code. You can see the code here and include it in your projects. 

If you liked this post, **please don't forget to upvote the [Kernel](https://www.kaggle.com/mlwhiz/third-place-model-for-toxic-comments-in-pytorch/edit) too.** I will be obliged.

## Endnotes and References

This post is a result of an effort of a lot of excellent Kagglers and I will try to reference them in this section. If I leave out someone, do understand that it was not my intention to do so.

- [Discussion on 3rd Place winner model in Toxic comment](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52644)
- [3rd Place model in Keras by Larry Freeman](https://www.kaggle.com/larryfreeman/toxic-comments-code-for-alexander-s-9872-model)
- [Pytorch starter Capsule model](https://www.kaggle.com/spirosrap/bilstm-attention-kfold-clr-extra-features-capsule)
- [How to: Preprocessing when using embeddings](https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings)
- [Improve your Score with some Text Preprocessing]( https://www.kaggle.com/theoviel/improve-your-score-with-some-text-preprocessing)
- [Pytorch baseline](https://www.kaggle.com/ziliwang/baseline-pytorch-bilstm)
- [Pytorch starter](https://www.kaggle.com/hengzheng/pytorch-starter)
