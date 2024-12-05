
---
title:   Understanding BERT with Huggingface
date:  2021-07-24
draft: false
url : blog/2021/07/24/huggingface-bert/
slug: huggingface-bert

Keywords:
- transformers from scratch
- create own transformer
- understanding transformers
- layman transformers
- layman transformers guide

Tags:
- Python
- Transformers
- Programming

Categories:
- Deep Learning
- Natural Language Processing
- Awesome Guides

description:

thumbnail : /images/huggingface-bert/main.png
image : /images/huggingface-bert/main.png
toc : false
type : "featured"
---


In my last post on [BERT](https://mlwhiz.medium.com/explaining-bert-simply-using-sketches-ba30f6f0c8cb), I talked in quite a detail about BERT transformers and how they work on a basic level. I went through the BERT Architecture, training data and training tasks.

But, as I like to say, we don’t really understand something before we implement it ourselves. So, in this post, we will implement a Question Answering Neural Network using BERT and HuggingFace Library.

---
## **What is a Question Answering Task?**

In this task, we are given a question and a paragraph in which the answer lies to our BERT Architecture and the objective is to determine the start and end span for the answer in the paragraph.

![Author Image: BERT Finetuning for Question-Answer Task](/images/huggingface-bert/0.png "Author Image: BERT Finetuning for Question-Answer Task")

As explained in the previous post, in the above example, we provide two inputs to the BERT architecture. The paragraph and the question separated by the <SEP> token. The purple layers are the output of the BERT encoder. We now define two vectors S and E(which will be learned during fine-tuning) both having shapes(1x768). We then get some scores by taking the dot product of these vectors and the second sentence’s output vectors. Once we have the scores, we can just apply SoftMax over all these scores to get probabilities. The training objective is the sum of the log-likelihoods of the correct start and end positions. Mathematically, for the Probability vector for Start positions:

![](/images/huggingface-bert/1.png)

where T_i is the word, we are focusing on. An analogous formula is for End positions.

To predict a span, we get all the scores — S.T and E.T and get the best span as the span having the maximum score, that is max(S.T_i + E.T_j) among all j≥i.

---
## How do we do this using Huggingface?

Simply, Huggingface provides a pretty straightforward way to do this.

```py
from datasets import load_dataset, load_metric
import random
from transformers import AutoTokenizer
import transformers
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
import torch
from transformers import default_data_collator
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
text = r"""
🤗 Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
TensorFlow 2.0 and PyTorch.
"""
questions = [
    "How many pretrained models are available in Transformers?",
    "What does Transformers provide?",
    "Transformers provides interoperability between which frameworks?",
]
for question in questions:
    inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]
text_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    pred = model(**inputs)
    answer_start_scores, answer_end_scores = pred['start_logits'][0] ,pred['end_logits'][0]

    answer_start = torch.argmax(
        answer_start_scores
    )  # Get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
print(f"Question: {question}")
    print(f"Answer: {answer}\n")

```

The output is:

    Question: How many pretrained models are available in Transformers?
    Answer: over 32 +

    Question: What does Transformers provide?
    Answer: general - purpose architectures

    Question: Transformers provides interoperability between which frameworks?
    Answer: TensorFlow 2. 0 and pytorch

So, here we just used the pretrained tokenizer and model on SQUAD dataset provided by Huggingface to get this done.
```py
tokenizer = AutoTokenizer.from_pretrained(“bert-large-uncased-whole-word-masking-finetuned-squad”)
model = AutoModelForQuestionAnswering.from_pretrained(“bert-large-uncased-whole-word-masking-finetuned-squad”)
```
Once we have the model we just get the start and end probability scores and predict the span as the one that lies between the token that has the maximum start score and the token that has the maximum end score.

So for example, if the start scores for paragraph are:

    ...
    Transformers - 0.1
    are - 0.2
    general - 0.5
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
    (BERT - 0.8
    ...

We will get the output as input_ids[answer_start:answer_end]where answer_start is the index of word general(one with max start score) and answer_end is index of (BERT(One with max end score).And the answer would be “general purpose architectures”.

---
## Fine-tuning Our Own Model using a Question-Answering dataset

Almost all the time we might want to train our own QA model on our own datasets. In that example, we will start from the SQUAD dataset and the base BERT Model in the Huggingface library to finetune it.

Lets look at how the SQUAD Dataset looks before we start finetuning the model

```py
datasets = load_dataset("squad")
def visualize(datasets, datatype = 'train', n_questions=10):
    n = len(datasets[datatype])
    random_questions=random.choices(list(range(n)),k=n_questions)
    for i in random_questions:
        print(f"Context:{datasets[datatype][i]['context']}")
        print(f"Question:{datasets[datatype][i]['question']}")
        print(f"Answer:{datasets[datatype][i]['answers']['text']}")
        print(f"Answer Start in Text:{datasets[datatype][i]['answers']['answer_start']}")
        print("-"*100)
visualize(datasets)
```

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

We can see each example contains the Context, Answer and the Start token for the Answer. We can use the script below to preprocess the data to the required format once we have the data in the above form. The script takes care of a lot of things amongst which the most important are the cases where the answer lies around `max_length` and calculating the span using the answer and the start token index.

```py
# Dealing with long docs:
max_length = 384 # The maximum length of a feature (question and context)
doc_stride = 128 # The authorized overlap between two part of the context when splitting it

def prepare_train_features(examples):
    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" ],
        examples["context" ],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    # Looks like [0,1,2,2,2,3,4,5,5...] - Here 2nd input pair has been split in 3 parts
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    # Looks like [[(0,0),(0,3),(3,4)...] ] - Contains the actual start indices and end indices for each word in the input.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples
  ```

Once we have data in required format we can just finetune our BERT base model.
```py
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
```

![](/images/huggingface-bert/2.png)

Once we train our model we can use it as:

```py
model = AutoModelForQuestionAnswering.from_pretrained("test-squad-trained")
text = r"""
🤗 Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
TensorFlow 2.0 and PyTorch
"""
questions = [
    "How many pretrained models are available in Transformers?",
    "What does Transformers provide?",
    "Transformers provides interoperability between which frameworks?",
]
for question in questions:
    inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]
text_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    pred = model(**inputs)
    answer_start_scores, answer_end_scores = pred['start_logits'][0] ,pred['end_logits'][0]

    answer_start = torch.argmax(
        answer_start_scores
    )  # Get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
print(f"Question: {question}")
    print(f"Answer: {answer}\n")

```

In this case also, we take the index of max start scores and max end scores and predict the answer as the one that is between. If we want to get the exact implementation as provided in the BERT Paper we can tweek the above code a little and find out the indexes which maximize `(start_score + end_score)`

```py
text = r"""
George Washington (February 22, 1732[b] – December 14, 1799) was an American political leader, military general, statesman, and Founding Father who served as the first president of the United States from 1789 to 1797. Previously, he led Patriot forces to victory in the nation's War for Independence. He presided at the Constitutional Convention of 1787, which established the U.S. Constitution and a federal government. Washington has been called the "Father of His Country" for his manifold leadership in the formative days of the new nation.
"""

question = "Who was the first president?"

inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")
input_ids = inputs["input_ids"].tolist()[0]

text_tokens = tokenizer.convert_ids_to_tokens(input_ids)

pred = model(**inputs)
answer_start_scores, answer_end_scores = pred['start_logits'][0] ,pred['end_logits'][0]

#get the index of first highest 20 tokens only
start_indexes = list(np.argsort(answer_start_scores.detach().numpy()))[::-1][:20]
end_indexes = list(np.argsort(answer_end_scores.detach().numpy()))[::-1][:20]

valid_answers = []
for start_index in start_indexes:
    for end_index in end_indexes:
        if start_index <= end_index:
            valid_answers.append(
                {
                    "score": answer_start_scores[start_index] + answer_end_scores[end_index],
                    "text": tokenizer.convert_tokens_to_string(
                        tokenizer.convert_ids_to_tokens(input_ids[start_index:end_index+1]))
                }
            )

valid_answers = sorted(valid_answers, key=lambda x: x["score"], reverse=True)

```

![](/images/huggingface-bert/3.png)

---
## References

* [Attention Is All You Need](https://arxiv.org/abs/1706.03762): The Paper which introduced Transformers.

* [BERT Paper](https://arxiv.org/abs/1810.04805): Do read this paper.

* [Huggingface](https://huggingface.co/transformers/usage.html)

In this post, I covered how we can create a Question Answering Model from scratch using BERT. I hope it would have been useful both for understanding BERT as well as Huggingface library.

If you want to look at other post in this series please take a look at:

* [Understanding Transformers, the Data Science Way](https://mlwhiz.com/blog/2020/09/20/transformers/)

* [Understanding Transformers, the Programming Way](https://mlwhiz.com/blog/2020/10/10/create-transformer-from-scratch/)

* [Explaining BERT Simply Using Sketches](https://mlwhiz.medium.com/explaining-bert-simply-using-sketches-ba30f6f0c8cb)

If you want to know more about NLP, I would like to recommend this awesome [Natural Language Processing Specialization](https://coursera.pxf.io/9WjZo0). You can start for free with the 7-day Free Trial. This course covers a wide range of tasks in Natural Language Processing from basic to advanced: sentiment analysis, summarization, dialogue state tracking, to name a few.

I am going to be writing more of such posts in the future too. Let me know what you think about them. Should I write on heavily technical topics or more beginner level articles? The comment section is your friend. Use it. Also, follow me up at [Medium](https://mlwhiz.medium.com/) or Subscribe to my [blog](mlwhiz.com).

And, finally a small disclaimer — There might be some affiliate links in this post to relevant resources, as sharing knowledge is never a bad idea.
