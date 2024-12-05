---
title:  Understanding Transformers, the Programming Way
date:  2020-10-10
draft: false
url : blog/2020/10/10/create-transformer-from-scratch/
slug: create-transformer-from-scratch

Keywords:
- Interview Preparation
- binary search

Tags:
- Python
- Programming

Categories:
- Programming
- Awesome Guides

description:

thumbnail : /images/create-transformer-from-scratch/main.png
image : /images/create-transformer-from-scratch/main.png
toc : false
type : "post"
---

Transformers have become the defacto standard for NLP tasks nowadays. They started being used in NLP but they are now being used in Computer Vision and sometimes to generate music as well. I am sure you would all have heard about the GPT3 Transformer or the jokes thereof.

***But everything aside, they are still hard to understand as ever.*** In my [last post](https://towardsdatascience.com/understanding-transformers-the-data-science-way-e4670a4ee076), I talked in quite a detail about transformers and how they work on a basic level. I went through the encoder and decoder architecture and the whole data flow in those different pieces of the neural network.

But as I like to say we don’t really understand something before we implement it ourselves. So in this post, we will implement an English to German language translator using Transformers.

---

## Task Description

We want to create a translator that uses transformers to convert English to German. So, if we look at it as a black-box, our network takes as input an English sentence and returns a German sentence.

![Transformer for Translation](/images/create-transformer-from-scratch/0.png "Transformer for Translation")

---

## Data Preprocessing

To train our English-German translation Model, we will need translated sentence pairs between English and German.

Fortunately, there is pretty much a standard way to get these with the IWSLT(International Workshop on Spoken Language Translation) dataset which we can access using `torchtext.datasets`. This machine translation dataset is sort of the defacto standard used for translation tasks and contains the translation of TED and TEDx talks on various topics in different languages.

Also, before we really get into the whole coding part, let us understand what we need as input and output to the model while training. We will actually need two matrices to be input to our Network:

![](/images/create-transformer-from-scratch/1.png)

* **The Source English sentences(Source):** A matrix of shape (batch size x source sentence length). The numbers in this matrix correspond to words based on the English vocabulary we will also need to create. So for example, 234 in the English vocabulary might correspond to the word “the”.  Also, do you notice that a lot of sentences end with a word whose index in vocabulary is 6? What is that about? Since all sentences don’t have the same length, they are padded with a word whose index is 6. So, 6 refers to `<blank>` token.

* **The Shifted Target German sentences(Target):** A matrix of shape (batch size x target sentence length). Here also the numbers in this matrix correspond to words based on the German vocabulary we will also need to create. If you notice that there seems to be a pattern to this particular matrix. All sentences start with a word whose index in german vocabulary is 2 and they invariably end with a pattern [3 and 0 or more 1's]. This is intentional as we want to start the target sentence with some start token(so 2 is for `<s>` token) and end the target sentence with some end token(so 3 is `</s>` token) and a string of blank tokens(so 1 refers to `<blank>` token). This part is covered more in detail in my last post on transformers, so if you are feeling confused here, I would ask you to take a look at that

So now as we know how to preprocess our data we will get into the actual code for preprocessing steps.

*Please note, that it really doesn’t matter here if you preprocess using other methods too. What eventually matters is that in the end, you need to send the sentence source and targets to your model in a way that's intended to be used by the transformer. i.e. source sentences should be padded with blank token and target sentences need to have a start token, an end token, and rest padded by blank tokens.*

We start by loading the Spacy Models which provides tokenizers to tokenize German and English text.

```py
# Load the Spacy Models
spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]
```

We also define some special tokens we will use for specifying blank/padding words, and beginning and end of sentences as discussed above.

```py
# Special Tokens
BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"
```

We can now define a preprocessing pipeline for both our source and target sentences using `data.field` from torchtext. You can notice that while we only specify pad_token for source sentence, we mention `pad_token`, `init_token` and `eos_token` for the target sentence. We also define which tokenizers to use.

```py
SRC = data.Field(tokenize=tokenize_en, pad_token=BLANK_WORD)
TGT = data.Field(tokenize=tokenize_de, init_token = BOS_WORD,
                 eos_token = EOS_WORD, pad_token=BLANK_WORD)
```

If you notice till now we haven’t seen any data. We now use IWSLT data from `torchtext.datasets` to create a train, validation, and test dataset. We also filter our sentences using the `MAX_LEN` parameter so that our code runs a lot faster. Notice that we are getting the data with `.en` and `.de` extensions. and we specify the preprocessing steps using the `fields` parameter.

```py
MAX_LEN = 20
train, val, test = datasets.IWSLT.splits(
    exts=('.en', '.de'), fields=(SRC, TGT),
    filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN
    and len(vars(x)['trg']) <= MAX_LEN)
```

So now since we have got our train data, let's see how it looks like:

```py
for i, example in enumerate([(x.src,x.trg) for x in train[0:5]]):
    print(f"Example_{i}:{example}")
```


    Example_0:(['David', 'Gallo', ':', 'This', 'is', 'Bill', 'Lange', '.', 'I', "'m", 'Dave', 'Gallo', '.'], ['David', 'Gallo', ':', 'Das', 'ist', 'Bill', 'Lange', '.', 'Ich', 'bin', 'Dave', 'Gallo', '.'])

    Example_1:(['And', 'we', "'re", 'going', 'to', 'tell', 'you', 'some', 'stories', 'from', 'the', 'sea', 'here', 'in', 'video', '.'], ['Wir', 'werden', 'Ihnen', 'einige', 'Geschichten', 'über', 'das', 'Meer', 'in', 'Videoform', 'erzählen', '.'])

    Example_2:(['And', 'the', 'problem', ',', 'I', 'think', ',', 'is', 'that', 'we', 'take', 'the', 'ocean', 'for', 'granted', '.'], ['Ich', 'denke', ',', 'das', 'Problem', 'ist', ',', 'dass', 'wir', 'das', 'Meer', 'für', 'zu', 'selbstverständlich', 'halten', '.'])

    Example_3:(['When', 'you', 'think', 'about', 'it', ',', 'the', 'oceans', 'are', '75', 'percent', 'of', 'the', 'planet', '.'], ['Wenn', 'man', 'darüber', 'nachdenkt', ',', 'machen', 'die', 'Ozeane', '75', '%', 'des', 'Planeten', 'aus', '.'])

    Example_4:(['Most', 'of', 'the', 'planet', 'is', 'ocean', 'water', '.'], ['Der', 'Großteil', 'der', 'Erde', 'ist', 'Meerwasser', '.'])

You might notice that while the `data.field` object has done the tokenization, it has not yet applied the start, end, and pad tokens and that is intentional. This is because we don’t have batches yet and the number of pad tokens will inherently depend on the maximum length of a sentence in the particular batch.

As mentioned in the start, we also create a Source and Target Language vocabulary by using the built-in function in `data.field` object. We specify a MIN_FREQ of 2 so that any word that doesn’t occur at least twice doesn’t get to be a part of our vocabulary.

```py
MIN_FREQ = 2
SRC.build_vocab(train.src, min_freq=MIN_FREQ)
TGT.build_vocab(train.trg, min_freq=MIN_FREQ)
```

Once we are done with this, we can simply use `data.Bucketiterator` which is used to giver batches of similar lengths to get our train iterator and validation iterator. Note that we use a `batch_size` of 1 for our validation data. It is optional to do this but done so that we don’t do padding or do minimal padding while checking validation data performance.

```py
BATCH_SIZE = 350

# Create iterators to process text in batches of approx. the same length by sorting on sentence lengths
train_iter = data.BucketIterator(train, batch_size=BATCH_SIZE, repeat=False, sort_key=lambda x: len(x.src))
val_iter = data.BucketIterator(val, batch_size=1, repeat=False, sort_key=lambda x: len(x.src))
```

Before we proceed, it is always a good idea to see how our batch looks like and what we are sending to the model as an input while training.

```py
batch = next(iter(train_iter))
src_matrix = batch.src.T
print(src_matrix, src_matrix.size())
```

This is our source matrix:

![](/images/create-transformer-from-scratch/2.png)

```py
trg_matrix = batch.trg.T
print(trg_matrix, trg_matrix.size())
```
And here is our target matrix:

![](/images/create-transformer-from-scratch/3.png)

So in the first batch, the `src_matrix` contains 350 sentences of length 20 and the `trg_matrix` is 350 sentences of length 22. Just so we are sure of our preprocessing, let’s see what some of these numbers represent in the `src_matrix` and the `trg_matrix`.

```py
print(SRC.vocab.itos[1])
print(TGT.vocab.itos[2])
print(TGT.vocab.itos[1])
```

    <blank>
    <s>
    <blank>

Just as expected. The opposite method, i.e. string to index also works well.
```py
print(TGT.vocab.stoi['</s>'])
```

    3

---

## The Transformer

![](/images/create-transformer-from-scratch/4.png)

So, now that we have a way to send the source sentence and the shifted target to our transformer, we can look at creating the Transformer.

```py
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MyTransformer(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu",source_vocab_length: int = 60000,target_vocab_length: int = 60000) -> None:
        super(MyTransformer, self).__init__()
        self.source_embedding = nn.Embedding(source_vocab_length, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.target_embedding = nn.Embedding(target_vocab_length, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self.out = nn.Linear(512, target_vocab_length)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        if src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")
        src = self.source_embedding(src)
        src = self.pos_encoder(src)
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        tgt = self.target_embedding(tgt)
        tgt = self.pos_encoder(tgt)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        output = self.out(output)
        return output


    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
```

A lot of the blocks here are taken from Pytorch `nn` module. Infact, Pytorch has a [Transformer](https://pytorch.org/docs/master/generated/torch.nn.Transformer.html) module too but it doesn’t include a lot of functionalities present in the paper like the embedding layer, and the PositionalEncoding layer. So this is sort of a more complete implementation that takes in a lot from pytorch implementation as well.

We create our Transformer particularly using these various blocks from Pytorch nn module:

* [TransformerEncoderLayer](https://pytorch.org/docs/master/generated/torch.nn.TransformerEncoderLayer.html): A single encoder layer

* [TransformerEncoder](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html): A stack of `num_encoder_layers` layers. In the paper, it is by default kept as 6.

* [TransformerDecoderLayer](https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoderLayer.html): A single decoder layer

* [TransformerDecoder](https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html): A stack of `num_decoder_layers` layers. In the paper it is by default kept as 6.

Also, note that whatever is happening in the layers is actually just matrix functions as I mentioned in the explanation post for transformers. See in particular how the decoder stack takes memory from encoder as input. We also create a positional encoding layer which lets us add the positional embedding to our word embedding.

If you want, you can look at the source code of all these blocks also which I have already linked. I had to look many times into the source code myself to make sure that I was giving the right inputs to these layers.

---
## Define Optimizer and Model

Now, we can initialize the transformer and the optimizer using:

```py
source_vocab_length = len(SRC.vocab)
target_vocab_length = len(TGT.vocab)
model = MyTransformer(source_vocab_length=source_vocab_length,target_vocab_length=target_vocab_length)
optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
model = model.cuda()
```
In the paper, the authors used an Adam optimizer with a scheduled learning rate but here I just using a normal Adam optimizer to keep things simple.

---
## Training our Translator

Now, we can train our transformer using the train function below. What we are necessarily doing in the training loop is:

* Getting the `src_matrix` and `trg_matrix` from a batch.

* Creating a `src_mask` — This is the mask that tells the model about the padded words in src_matrix data.

* Creating a `trg_mask` — So that our model is not able to look at the future subsequent target words at any point in time.

* Getting the prediction from the model.

* Calculating loss using cross-entropy. (In the paper they use KL divergence, but this also works fine for understanding)

* Backprop.

* We save the best model based on validation loss.

* We also predict the model output at every epoch for some sentences of our choice as a debug step using the function `greedy_decode_sentence`. We will discuss this function in the results section.

```py
def train(train_iter, val_iter, model, optim, num_epochs,use_gpu=True):
    train_losses = []
    valid_losses = []
    for epoch in range(num_epochs):
        train_loss = 0
        valid_loss = 0
        # Train model
        model.train()
        for i, batch in enumerate(train_iter):
            src = batch.src.cuda() if use_gpu else batch.src
            trg = batch.trg.cuda() if use_gpu else batch.trg
            #change to shape (bs , max_seq_len)
            src = src.transpose(0,1)
            #change to shape (bs , max_seq_len+1) , Since right shifted
            trg = trg.transpose(0,1)
            trg_input = trg[:, :-1]
            targets = trg[:, 1:].contiguous().view(-1)
            src_mask = (src != 0)
            src_mask = src_mask.float().masked_fill(src_mask == 0, float('-inf')).masked_fill(src_mask == 1, float(0.0))
            src_mask = src_mask.cuda() if use_gpu else src_mask
            trg_mask = (trg_input != 0)
            trg_mask = trg_mask.float().masked_fill(trg_mask == 0, float('-inf')).masked_fill(trg_mask == 1, float(0.0))
            trg_mask = trg_mask.cuda() if use_gpu else trg_mask
            size = trg_input.size(1)
            #print(size)
            np_mask = torch.triu(torch.ones(size, size)==1).transpose(0,1)
            np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0.0))
            np_mask = np_mask.cuda() if use_gpu else np_mask
            # Forward, backprop, optimizer
            optim.zero_grad()
            preds = model(src.transpose(0,1), trg_input.transpose(0,1), tgt_mask = np_mask)#, src_mask = src_mask)#, tgt_key_padding_mask=trg_mask)
            preds = preds.transpose(0,1).contiguous().view(-1, preds.size(-1))
            loss = F.cross_entropy(preds,targets, ignore_index=0,reduction='sum')
            loss.backward()
            optim.step()
            train_loss += loss.item()/BATCH_SIZE

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_iter):
                src = batch.src.cuda() if use_gpu else batch.src
                trg = batch.trg.cuda() if use_gpu else batch.trg
                #change to shape (bs , max_seq_len)
                src = src.transpose(0,1)
                #change to shape (bs , max_seq_len+1) , Since right shifted
                trg = trg.transpose(0,1)
                trg_input = trg[:, :-1]
                targets = trg[:, 1:].contiguous().view(-1)
                src_mask = (src != 0)
                src_mask = src_mask.float().masked_fill(src_mask == 0, float('-inf')).masked_fill(src_mask == 1, float(0.0))
                src_mask = src_mask.cuda() if use_gpu else src_mask
                trg_mask = (trg_input != 0)
                trg_mask = trg_mask.float().masked_fill(trg_mask == 0, float('-inf')).masked_fill(trg_mask == 1, float(0.0))
                trg_mask = trg_mask.cuda() if use_gpu else trg_mask
                size = trg_input.size(1)
                #print(size)
                np_mask = torch.triu(torch.ones(size, size)==1).transpose(0,1)
                np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0.0))
                np_mask = np_mask.cuda() if use_gpu else np_mask

                preds = model(src.transpose(0,1), trg_input.transpose(0,1), tgt_mask = np_mask)#, src_mask = src_mask)#, tgt_key_padding_mask=trg_mask)
                preds = preds.transpose(0,1).contiguous().view(-1, preds.size(-1))
                loss = F.cross_entropy(preds,targets, ignore_index=0,reduction='sum')
                valid_loss += loss.item()/1

        # Log after each epoch
        print(f'''Epoch [{epoch+1}/{num_epochs}] complete. Train Loss: {train_loss/len(train_iter):.3f}. Val Loss: {valid_loss/len(val_iter):.3f}''')

        #Save best model till now:
        if valid_loss/len(val_iter)<min(valid_losses,default=1e9):
            print("saving state dict")
            torch.save(model.state_dict(), f"checkpoint_best_epoch.pt")

        train_losses.append(train_loss/len(train_iter))
        valid_losses.append(valid_loss/len(val_iter))

        # Check Example after each epoch:
        sentences = ["This is an example to check how our model is performing."]
        for sentence in sentences:
            print(f"Original Sentence: {sentence}")
            print(f"Translated Sentence: {greeedy_decode_sentence(model,sentence)}")
    return train_losses,valid_losses
```

We can now run our training using:
```py
train_losses,valid_losses = train(train_iter, val_iter, model, optim, 35)
```
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

---
## Results

We can plot the training and validation losses using Plotly express.
```py
import pandas as pd
import plotly.express as px
losses = pd.DataFrame({'train_loss':train_losses,'val_loss':valid_losses})
px.line(losses,y = ['train_loss','val_loss'])
```

![](/images/create-transformer-from-scratch/5.png)

If we want to deploy this model we can load it simply using:
```py
model.load_state_dict(torch.load(f”checkpoint_best_epoch.pt”))
```
and predict for any source sentence using the `greeedy_decode_sentence` function, which is:

```py
def greeedy_decode_sentence(model,sentence):
    model.eval()
    sentence = SRC.preprocess(sentence)
    indexed = []
    for tok in sentence:
        if SRC.vocab.stoi[tok] != 0 :
            indexed.append(SRC.vocab.stoi[tok])
        else:
            indexed.append(0)
    sentence = Variable(torch.LongTensor([indexed])).cuda()
    trg_init_tok = TGT.vocab.stoi[BOS_WORD]
    trg = torch.LongTensor([[trg_init_tok]]).cuda()
    translated_sentence = ""
    maxlen = 25
    for i in range(maxlen):
        size = trg.size(0)
        np_mask = torch.triu(torch.ones(size, size)==1).transpose(0,1)
        np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0.0))
        np_mask = np_mask.cuda()
        pred = model(sentence.transpose(0,1), trg, tgt_mask = np_mask)
        add_word = TGT.vocab.itos[pred.argmax(dim=2)[-1]]
        translated_sentence+=" "+add_word
        if add_word==EOS_WORD:
            break
        trg = torch.cat((trg,torch.LongTensor([[pred.argmax(dim=2)[-1]]]).cuda()))
        #print(trg)
    return translated_sentence
```

![Predicting with a greedy search using the Transformer](/images/create-transformer-from-scratch/6.gif "Predicting with a greedy search using the Transformer")

This function does piecewise predictions. The greedy search would start with:

* Passing the whole English sentence as encoder input and just the start token `<s>` as shifted output(input to the decoder) to the model and doing the forward pass.

* The model will predict the next word — `der`

* Then, we pass the whole English sentence as encoder input and add the last predicted word to the shifted output(input to the decoder = `<s> der`) and do the forward pass.

* The model will predict the next word — `schnelle`

* Passing the whole English sentence as encoder input and `<s> der schnelle` as shifted output(input to the decoder) to the model and doing the forward pass.

* and so on, until the model predicts the end token `</s>` or we generate some maximum number of tokens(something we can define) so the translation doesn’t run for an infinite duration in any case it breaks.

Now we can translate any sentence using this:

```py
sentence = "Isn't Natural language processing just awesome? Please do let me know in the comments."
print(greeedy_decode_sentence(model,sentence))
```

    Ist es nicht einfach toll ? Bitte lassen Sie mich gerne in den Kommentare kennen . </s>

Since I don’t have a German Translator at hand, I will use the next best thing to see how our model is performing. Let us take help of google translate service to understand what this german sentence means.

![](/images/create-transformer-from-scratch/7.png)

There seem to be some mistakes in the translation as “Natural Language Processing” is not there(Ironic?) but it seems like a good enough translation to me as the neural network is somehow able to understand the structure of both the languages with just an hour of training.

---

## Vorbehalte / Verbesserungen (Caveats/Improvements)

We might have achieved better results if we did everything in the same way the paper did:

* Train on whole data

* Byte Pair Encoding

* Learning Rate Scheduling

* KL Divergence Loss

* Beam search, and

* Checkpoint ensembling

I discussed all of these in my [last post](https://towardsdatascience.com/understanding-transformers-the-data-science-way-e4670a4ee076) and all of these are easy to implement additions. But this simple implementation was meant to understand how a transformer works so I didn’t include all these so as not to confuse the readers. There have actually been quite a lot of advancements on top of transformers that have allowed us to have much better models for translation. We will discuss those advancements and how they came about in the upcoming post, where I will talk about BERT, one of the most popular NLP models that utilizes a Transformer at its core.

---

## References

* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

* [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)

In this post, We created an English to German translation network almost from scratch using the transformer architecture.

For a closer look at the code for this post, please visit my [GitHub](https://github.com/MLWhiz/data_science_blogs/tree/master/transformers) repository where you can find the code for this post as well as all my posts.

**As a side note**: If you want to know more about NLP, I would like to recommend this awesome [Natural Language Processing Specialization](https://coursera.pxf.io/9WjZo0). You can start for free with the 7-day Free Trial. This course covers a wide range of tasks in Natural Language Processing from basic to advanced: sentiment analysis, summarization, dialogue state tracking, to name a few.

I am going to be writing more of such posts in the future too. Let me know what you think about them. Should I write on heavily technical topics or more beginner level articles? The comment section is your friend. Use it. Also, follow me up at [Medium](https://mlwhiz.medium.com/) or Subscribe to my [blog](mlwhiz.com).

And, finally a small disclaimer — There might be some affiliate links in this post to relevant resources, as sharing knowledge is never a bad idea.

This post was first published [here](https://lionbridge.ai/articles/transformers-in-nlp-creating-a-translator-model-from-scratch/)
