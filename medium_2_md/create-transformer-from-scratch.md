
# Understanding Transformers, the Programming Way

Understanding Transformers, the Programming Way

### Because you can only understand it, if you can program it

Transformers have become the defacto standard for NLP tasks nowadays. They started being used in NLP but they are now being used in Computer Vision and sometimes to generate music as well. I am sure you would all have heard about the GPT3 Transformer or the jokes thereof.

***But everything aside, they are still hard to understand as ever. ***In my last post, I talked in quite a detail about transformers and how they work on a basic level. I went through the encoder and decoder architecture and the whole data flow in those different pieces of the neural network. 

But as I like to say we don’t really understand something before we implement it ourselves. So in this post, we will implement an English to German language translator using Transformers.

## Task Description

We want to create a translator that uses transformers to convert English to German. So, if we look at it as a black-box, our network takes as input an English sentence and returns a German sentence.

![Transformer for Translation](https://cdn-images-1.medium.com/max/3366/1*NBCtrY02DTg9ZQiK7dxaKQ.png)*Transformer for Translation*

## Data Preprocessing

To train our English-German translation Model, we will need translated sentence pairs between English and German. 

Fortunately, there is pretty much a standard way to get these with the IWSLT(International Workshop on Spoken Language Translation) dataset which we can access using torchtext.datasets. This machine translation dataset is sort of the defacto standard used for translation tasks and contains the translation of TED and TEDx talks on various topics in different languages.

Also, before we really get into the whole coding part, let us understand what we need as input and output to the model while training. We will actually need two matrices to be input to our Network:

![](https://cdn-images-1.medium.com/max/3246/1*UddDCpMGLBIz33AQZtaEZg.png)

* **The Source English sentences(Source):** A matrix of shape (batch size x source sentence length). The numbers in this matrix correspond to words based on the English vocabulary we will also need to create. So for example, 234 in the English vocabulary might correspond to the word “the”.  Also, do you notice that a lot of sentences end with a word whose index in vocabulary is 6? What is that about? Since all sentences don’t have the same length, they are padded with a word whose index is 6. So, 6 refers to <blank> token.

* **The Shifted Target German sentences(Target):** A matrix of shape (batch size x target sentence length). Here also the numbers in this matrix correspond to words based on the German vocabulary we will also need to create. If you notice that there seems to be a pattern to this particular matrix. All sentences start with a word whose index in german vocabulary is 2 and they invariably end with a pattern [3 and 0 or more 1's]. This is intentional as we want to start the target sentence with some start token(so 2 is for <s> token) and end the target sentence with some end token(so 3 is </s> token) and a string of blank tokens(so 1 refers to <blank> token). This part is covered more in detail in my last post on transformers, so if you are feeling confused here, I would ask you to take a look at that

So now as we know how to preprocess our data we will get into the actual code for preprocessing steps. 

*Please note, that it really doesn’t matter here if you preprocess using other methods too. What eventually matters is that in the end, you need to send the sentence source and targets to your model in a way that's intended to be used by the transformer. i.e. source sentences should be padded with blank token and target sentences need to have a start token, an end token, and rest padded by blank tokens.*

We start by loading the Spacy Models which provides tokenizers to tokenize German and English text.

    # Load the Spacy Models
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

We also define some special tokens we will use for specifying blank/padding words, and beginning and end of sentences as discussed above.

    # Special Tokens
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"

We can now define a preprocessing pipeline for both our source and target sentences using data.field from torchtext. You can notice that while we only specify pad_token for source sentence, we mention pad_token, init_token and eos_token for the target sentence. We also define which tokenizers to use. 

    SRC = data.Field(tokenize=tokenize_en, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_de, init_token = BOS_WORD, 
                     eos_token = EOS_WORD, pad_token=BLANK_WORD)

If you notice till now we haven’t seen any data. We now use IWSLT data from torchtext.datasets to create a train, validation, and test dataset. We also filter our sentences using the MAX_LEN parameter so that our code runs a lot faster. Notice that we are getting the data with .en and .de extensions. and we specify the preprocessing steps using the fields parameter.

    MAX_LEN = 20
    train, val, test = datasets.IWSLT.splits(
        exts=('.en', '.de'), fields=(SRC, TGT), 
        filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN 
        and len(vars(x)['trg']) <= MAX_LEN)

So now since we have got our train data, let's see how it looks like:

    for i, example in enumerate([(x.src,x.trg) for x in train[0:5]]):
        print(f"Example_{i}:{example}")

    ---------------------------------------------------------------

    Example_0:(['David', 'Gallo', ':', 'This', 'is', 'Bill', 'Lange', '.', 'I', "'m", 'Dave', 'Gallo', '.'], ['David', 'Gallo', ':', 'Das', 'ist', 'Bill', 'Lange', '.', 'Ich', 'bin', 'Dave', 'Gallo', '.'])

    Example_1:(['And', 'we', "'re", 'going', 'to', 'tell', 'you', 'some', 'stories', 'from', 'the', 'sea', 'here', 'in', 'video', '.'], ['Wir', 'werden', 'Ihnen', 'einige', 'Geschichten', 'über', 'das', 'Meer', 'in', 'Videoform', 'erzählen', '.'])

    Example_2:(['And', 'the', 'problem', ',', 'I', 'think', ',', 'is', 'that', 'we', 'take', 'the', 'ocean', 'for', 'granted', '.'], ['Ich', 'denke', ',', 'das', 'Problem', 'ist', ',', 'dass', 'wir', 'das', 'Meer', 'für', 'zu', 'selbstverständlich', 'halten', '.'])

    Example_3:(['When', 'you', 'think', 'about', 'it', ',', 'the', 'oceans', 'are', '75', 'percent', 'of', 'the', 'planet', '.'], ['Wenn', 'man', 'darüber', 'nachdenkt', ',', 'machen', 'die', 'Ozeane', '75', '%', 'des', 'Planeten', 'aus', '.'])

    Example_4:(['Most', 'of', 'the', 'planet', 'is', 'ocean', 'water', '.'], ['Der', 'Großteil', 'der', 'Erde', 'ist', 'Meerwasser', '.'])

You might notice that while the data.field object has done the tokenization, it has not yet applied the start, end, and pad tokens and that is intentional. This is because we don’t have batches yet and the number of pad tokens will inherently depend on the maximum length of a sentence in the particular batch.

As mentioned in the start, we also create a Source and Target Language vocabulary by using the built-in function in data.field object. We specify a MIN_FREQ of 2 so that any word that doesn’t occur at least twice doesn’t get to be a part of our vocabulary.

    MIN_FREQ = 2
    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

Once we are done with this, we can simply use data.Bucketiterator which is used to giver batches of similar lengths to get our train iterator and validation iterator. Note that we use a batch_size of 1 for our validation data. It is optional to do this but done so that we don’t do padding or do minimal padding while checking validation data performance.

    BATCH_SIZE = 350

    # Create iterators to process text in batches of approx. the same length by sorting on sentence lengths

    train_iter = data.BucketIterator(train, batch_size=BATCH_SIZE, repeat=False, sort_key=lambda x: len(x.src))

    val_iter = data.BucketIterator(val, batch_size=1, repeat=False, sort_key=lambda x: len(x.src))

Before we proceed, it is always a good idea to see how our batch looks like and what we are sending to the model as an input while training. 

    batch = next(iter(train_iter))
    src_matrix = batch.src.T
    print(src_matrix, src_matrix.size())

This is our source matrix:

![](https://cdn-images-1.medium.com/max/2000/1*X-TGYIgyk8mgyiuhhX99sQ.png)

    trg_matrix = batch.trg.T
    print(trg_matrix, trg_matrix.size())

And here is our target matrix:

![](https://cdn-images-1.medium.com/max/2000/1*8S6-VEks0t2cWYwHikCnQw.png)

So in the first batch, the src_matrix contains 350 sentences of length 20 and the trg_matrix is 350 sentences of length 22. Just so we are sure of our preprocessing, let’s see what some of these numbers represent in the src_matrix and the trg_matrix. 

    print(SRC.vocab.itos[1])
    print(TGT.vocab.itos[2])
    print(TGT.vocab.itos[1])
    --------------------------------------------------------------------
    <blank>
    <s>
    <blank>

Just as expected. The opposite method, i.e. string to index also works well. 

    print(TGT.vocab.stoi['</s>'])
    --------------------------------------------------------------------
    3

## The Transformer

![](https://cdn-images-1.medium.com/max/4410/1*a3orJQUKYY8xzTWJK1cjKQ.png)

So, now that we have a way to send the source sentence and the shifted target to our transformer, we can look at creating the Transformer. 

<iframe src="https://medium.com/media/e6a07e6e3364d16ddd73142520b82b11" frameborder=0></iframe>

A lot of the blocks here are taken from Pytorch nn module. Infact, Pytorch has a [Transformer](https://pytorch.org/docs/master/generated/torch.nn.Transformer.html) module too but it doesn’t include a lot of functionalities present in the paper like the embedding layer, and the PositionalEncoding layer. So this is sort of a more complete implementation that takes in a lot from pytorch implementation as well. 

We create our Transformer particularly using these various blocks from Pytorch nn module:

* [TransformerEncoderLayer](https://pytorch.org/docs/master/generated/torch.nn.TransformerEncoderLayer.html): A single encoder layer

* [TransformerEncoder](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html): A stack of num_encoder_layers layers. In the paper, it is by default kept as 6.

* [TransformerDecoderLayer](https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoderLayer.html): A single decoder layer

* [TransformerDecoder](https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html): A stack of num_decoder_layers layers. In the paper it is by default kept as 6.

Also, note that whatever is happening in the layers is actually just matrix functions as I mentioned in the explanation post for transformers. See in particular how the decoder stack takes memory from encoder as input. We also create a positional encoding layer which lets us add the positional embedding to our word embedding.

If you want, you can look at the source code of all these blocks also which I have already linked. I had to look many times into the source code myself to make sure that I was giving the right inputs to these layers.

## Define Optimizer and Model

Now, we can initialize the transformer and the optimizer using:

    source_vocab_length = len(SRC.vocab)
    target_vocab_length = len(TGT.vocab)

    model = MyTransformer(source_vocab_length=source_vocab_length,target_vocab_length=target_vocab_length)

    optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    model = model.cuda()

In the paper, the authors used an Adam optimizer with a scheduled learning rate but here I just using a normal Adam optimizer to keep things simple.

## Training our Translator

Now, we can train our transformer using the train function below. What we are necessarily doing in the training loop is:

* Getting the src_matrix and trg_matrix from a batch. 

* Creating a src_mask — This is the mask that tells the model about the padded words in src_matrix data.

* Creating a trg_mask — So that our model is not able to look at the future subsequent target words at any point in time.

* Getting the prediction from the model.

* Calculating loss using cross-entropy. (In the paper they use KL divergence, but this also works fine for understanding) 

* Backprop.

* We save the best model based on validation loss.

* We also predict the model output at every epoch for some sentences of our choice as a debug step using the function greedy_decode_sentence. We will discuss this function in the results section.

<iframe src="https://medium.com/media/5ba1d017d6b9e9a069dade257919c96c" frameborder=0></iframe>

We can now run our training using:

    train_losses,valid_losses = train(train_iter, val_iter, model, optim, 35)

Below is the output of the training loop (shown only for some epochs):

    **Epoch [1/35] complete.** Train Loss: 86.092. Val Loss: 64.514
    Original Sentence: This is an example to check how our model is performing.
    Translated Sentence:  Und die der der der der der der der der der der der der der der der der der der der der der der der

    **Epoch [2/35] complete.** Train Loss: 59.769. Val Loss: 55.631
    Original Sentence: This is an example to check how our model is performing.
    Translated Sentence:  Das ist ein paar paar paar sehr , die das ist ein paar sehr Jahre . </s>

    .
    .
    .
    .

    **Epoch [16/35] complete.** Train Loss: 21.791. Val Loss: 28.000
    Original Sentence: This is an example to check how our model is performing.
    Translated Sentence:  Hier ist ein Beispiel , um zu prüfen , wie unser Modell aussieht . Das ist ein Modell . </s>

    .
    .
    .
    .

    **Epoch [34/35] complete.** Train Loss: 9.492. Val Loss: 31.005
    Original Sentence: This is an example to check how our model is performing.
    Translated Sentence:  Hier ist ein Beispiel , um prüfen zu überprüfen , wie unser Modell ist . Wir spielen . </s>

    **Epoch [35/35] complete.** Train Loss: 9.014. Val Loss: 32.097
    Original Sentence: This is an example to check how our model is performing.
    Translated Sentence:  Hier ist ein Beispiel , um prüfen wie unser Modell ist . Wir spielen . </s>

We can see how our model starts with a gibberish translation — “Und die der der der der der der der der der der der der der der der der der der der der der der der” and starts giving us something by the end of a few iterations. 

## Results

We can plot the training and validation losses using Plotly express.

    import pandas as pd
    import plotly.express as px

    losses = pd.DataFrame({'train_loss':train_losses,'val_loss':valid_losses})

    px.line(losses,y = ['train_loss','val_loss'])

![](https://cdn-images-1.medium.com/max/2514/1*--mwu9F7t4uI3eQn__Gmhw.png)

If we want to deploy this model we can load it simply using:

    model.load_state_dict(torch.load(f”checkpoint_best_epoch.pt”))

and predict for any source sentence using the greeedy_decode_sentence function, which is:

<iframe src="https://medium.com/media/096cc6762cd0482d911f9e9629dad91e" frameborder=0></iframe>

![Predicting with a greedy search using the Transformer](https://cdn-images-1.medium.com/max/3020/1*sdjUHzfL9arBGjMModT9vw.gif)*Predicting with a greedy search using the Transformer*

This function does piecewise predictions. The greedy search would start with:

* Passing the whole English sentence as encoder input and just the start token <s> as shifted output(input to the decoder) to the model and doing the forward pass.

* The model will predict the next word — der

* Then, we pass the whole English sentence as encoder input and add the last predicted word to the shifted output(input to the decoder = <s> der) and do the forward pass.

* The model will predict the next word — schnelle

* Passing the whole English sentence as encoder input and <s> der schnelle as shifted output(input to the decoder) to the model and doing the forward pass.

* and so on, until the model predicts the end token </s> or we generate some maximum number of tokens(something we can define) so the translation doesn’t run for an infinite duration in any case it breaks.

Now we can translate any sentence using this:

    sentence = "Isn't Natural language processing just awesome? Please do let me know in the comments."

    print(greeedy_decode_sentence(model,sentence))

    ------------------------------------------------------------------

    Ist es nicht einfach toll ? Bitte lassen Sie mich gerne in den Kommentare kennen . </s>

Since I don’t have a German Translator at hand, I will use the next best thing to see how our model is performing. Let us take help of google translate service to understand what this german sentence means. 

![](https://cdn-images-1.medium.com/max/2828/1*D6quLQHvfRyWDdTJ_ChibA.png)

There seem to be some mistakes in the translation as “Natural Language Processing” is not there(Ironic?) but it seems like a good enough translation to me as the neural network is somehow able to understand the structure of both the languages with just an hour of training.

## Vorbehalte / Verbesserungen (Caveats/Improvements)

We might have achieved better results if we did everything in the same way the paper did: 

* Train on whole data

* Byte Pair Encoding 

* Learning Rate Scheduling

* KL Divergence Loss

* Beam search, and

* Checkpoint ensembling

I discussed all of these in my last post and all of these are easy to implement additions. But this simple implementation was meant to understand how a transformer works so I didn’t include all these so as not to confuse the readers. There have actually been quite a lot of advancements on top of transformers that have allowed us to have much better models for translation. We will discuss those advancements and how they came about in the upcoming post, where I will talk about BERT, one of the most popular NLP models that utilizes a Transformer at its core. 

## References

* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

* [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)

In this post, We created an English to German translation network almost from scratch using the transformer architecture. 

For a closer look at the code for this post, please visit my [GitHub](https://github.com/MLWhiz/data_science_blogs/tree/master/transformers) repository where you can find the code for this post as well as all my posts.

If you want to learn more about NLP, I would like to call out an excellent course on [**Natural Language Processing](https://click.linksynergy.com/link?id=lVarvwc5BD0&offerid=467035.11503135394&type=2&murl=https%3A%2F%2Fwww.coursera.org%2Flearn%2Flanguage-processing)** from the Advanced Machine Learning Specialization. Do check it out.

I am going to be writing more of such posts in the future too. Let me know what you think about them. Should I write on heavily technical topics or more beginner level articles? The comment section is your friend. Use it. Also, follow me up at [**Medium](https://medium.com/@rahul_agarwal)** or Subscribe to my [**blog](https://mlwhiz.ck.page/a9b8bda70c)**.

And, finally a small disclaimer — There might be some affiliate links in this post to relevant resources, as sharing knowledge is never a bad idea.
