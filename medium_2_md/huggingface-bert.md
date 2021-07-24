
# Understanding BERT with Huggingface

Understanding BERT with Huggingface

### Using BERT and Huggingface to create a Question Answer Model

In my last post on [BERT](https://mlwhiz.medium.com/explaining-bert-simply-using-sketches-ba30f6f0c8cb), I talked in quite a detail about BERT transformers and how they work on a basic level. I went through the BERT Architecture, training data and training tasks.

But, as I like to say, we don’t really understand something before we implement it ourselves. So, in this post, we will implement a Question Answering Neural Network using BERT and HuggingFace Library.

## **What is a Question Answering Task?**

In this task, we are given a question and a paragraph in which the answer lies to our BERT Architecture and the objective is to determine the start and end span for the answer in the paragraph.

![Author Image: BERT Finetuning for Question-Answer Task](https://cdn-images-1.medium.com/max/5200/0*0e3UKmulEFwGO76U.png)*Author Image: BERT Finetuning for Question-Answer Task*

As explained in the previous post, in the above example, we provide two inputs to the BERT architecture. The paragraph and the question separated by the <SEP> token. The purple layers are the output of the BERT encoder. We now define two vectors S and E(which will be learned during fine-tuning) both having shapes(1x768). We then get some scores by taking the dot product of these vectors and the second sentence’s output vectors. Once we have the scores, we can just apply SoftMax over all these scores to get probabilities. The training objective is the sum of the log-likelihoods of the correct start and end positions. Mathematically, for the Probability vector for Start positions:

![](https://cdn-images-1.medium.com/max/2000/0*7kpsEpvSXQeORh5B.png)

where T_i is the word, we are focusing on. An analogous formula is for End positions.

To predict a span, we get all the scores — S.T and E.T and get the best span as the span having the maximum score, that is max(S.T_i + E.T_j) among all j≥i.

## How do we do this using Huggingface?

Simply, Huggingface provides a pretty straightforward way to do this.

<iframe src="https://medium.com/media/2c25ca756ad2fa8b5040c5b04e48d79f" frameborder=0></iframe>

The output is:

    Question: How many pretrained models are available in Transformers?
    Answer: over 32 +
    
    Question: What does Transformers provide?
    Answer: general - purpose architectures
    
    Question: Transformers provides interoperability between which frameworks?
    Answer: TensorFlow 2. 0 and pytorch

So, here we just used the pretrained tokenizer and model on SQUAD dataset provided by Huggingface to get this done.

    tokenizer = AutoTokenizer.from_pretrained(“bert-large-uncased-whole-word-masking-finetuned-squad”)
    model = AutoModelForQuestionAnswering.from_pretrained(“bert-large-uncased-whole-word-masking-finetuned-squad”)

Once we have the model we just get the start and end probability scores and predict the span as the one that lies between the token that has the maximum start score and the token that has the maximum end score.

So for example, if the start scores for paragraph are:

    ...
    Transformers - 0.1
    are - 0.2
    **general - 0.5**
    purpose - 0.1
    architectures -0.01
    (BERT - 0.001
    ...

And the end scores are:

    ...
    Transformers - 0.01
    are - 0.02
    general - 0.05
    purpose - 0.01
    architectures -0.01
    **(BERT - 0.8**
    ...

We will get the output as input_ids[answer_start:answer_end]where answer_start is the index of word general(one with max start score) and answer_end is index of (BERT(One with max end score).And the answer would be “general purpose architectures”.

## Fine-tuning Our Own Model using a Question-Answering dataset

Almost all the time we might want to train our own QA model on our own datasets. In that example, we will start from the SQUAD dataset and the base BERT Model in the Huggingface library to finetune it.

Lets look at how the SQUAD Dataset looks before we start finetuning the model

<iframe src="https://medium.com/media/70ae32a0750b11dc4d8db60822d20b7a" frameborder=0></iframe>

    Context:Within computer systems, two of many security models capable of enforcing privilege separation are access control lists (ACLs) and capability-based security. Using ACLs to confine programs has been proven to be insecure in many situations, such as if the host computer can be tricked into indirectly allowing restricted file access, an issue known as the confused deputy problem. It has also been shown that the promise of ACLs of giving access to an object to only one person can never be guaranteed in practice. Both of these problems are resolved by capabilities. This does not mean practical flaws exist in all ACL-based systems, but only that the designers of certain utilities must take responsibility to ensure that they do not introduce flaws.[citation needed]
    Question:The confused deputy problem and the problem of not guaranteeing only one person has access are resolved by what?
    Answer:['capabilities']
    Answer Start in Text:[553]
    --------------------------------------------------------------------
    Context:In recent years, the nightclubs on West 27th Street have succumbed to stiff competition from Manhattan's Meatpacking District about fifteen blocks south, and other venues in downtown Manhattan.
    Question:How many blocks south of 27th Street is Manhattan's Meatpacking District?
    Answer:['fifteen blocks']
    Answer Start in Text:[132]
    --------------------------------------------------------------------

We can see each example contains the Context, Answer and the Start token for the Answer. We can use the script below to preprocess the data to the required format once we have the data in the above form. The script takes care of a lot of things amongst which the most important are the cases where the answer lies around max_length and calculating the span using the answer and the start token index.

<iframe src="https://medium.com/media/4719be3f7ffc16ee48a87d78b9951e98" frameborder=0></iframe>

Once we have data in required format we can just finetune our BERT base model.

    model_checkpoint = "bert-base-uncased"
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    tokenized_datasets = datasets.map(prepare_train_features, batched=True, remove_columns=datasets["train"].column_names)

    args = TrainingArguments(
        f"test-squad",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    data_collator = default_data_collator
    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(trainer.save_model("test-squad-trained"))

![](https://cdn-images-1.medium.com/max/2016/1*vZPc1zLYPm9N50kIjCl5Ig.png)

Once we train our model we can use it as:

<iframe src="https://medium.com/media/d008f352477f30c9cc29fc2f62e683ab" frameborder=0></iframe>

In this case also, we take the index of max start scores and max end scores and predict the answer as the one that is between. If we want to get the exact implementation as provided in the BERT Paper we can tweek the above code a little and find out the indexes which maximize (start_score + end_score)

<iframe src="https://medium.com/media/fd5722a8e7e5f81a8e48b0c202bd7d21" frameborder=0></iframe>

![](https://cdn-images-1.medium.com/max/3668/1*QRyhOsJXm022Kkurcf0OBQ.png)

## References

* [Attention Is All You Need](https://arxiv.org/abs/1706.03762): The Paper which introduced Transformers.

* [BERT Paper](https://arxiv.org/abs/1810.04805): Do read this paper.

* [Huggingface](https://huggingface.co/transformers/usage.html)

In this post, I covered how we can create a Question Answering Model from scratch using BERT. I hope it would have been useful both for understanding BERT as well as Huggingface library.

If you want to look at other post in this series please take a look at:

* [Understanding Transformers, the Data Science Way](https://mlwhiz.com/blog/2020/09/20/transformers/)

* [Understanding Transformers, the Programming Way](https://mlwhiz.com/blog/2020/10/10/create-transformer-from-scratch/)

* [Explaining BERT Simply Using Sketches](https://mlwhiz.medium.com/explaining-bert-simply-using-sketches-ba30f6f0c8cb)

If you want to learn more about NLP, I would like to call out an excellent [Natural Language Processing](https://click.linksynergy.com/link?id=lVarvwc5BD0&offerid=467035.11503135394&type=2&murl=https%3A%2F%2Fwww.coursera.org%2Flearn%2Flanguage-processing) course from the Advanced Machine Learning Specialization. Do check it out.

I am going to be writing more of such posts in the future too. Let me know what you think about them. Should I write on heavily technical topics or more beginner level articles? The comment section is your friend. Use it. Also, follow me up at [Medium](https://mlwhiz.medium.com/) or Subscribe to my [blog](https://mlwhiz.ck.page/a9b8bda70c).

And, finally a small disclaimer — There might be some affiliate links in this post to relevant resources, as sharing knowledge is never a bad idea.
