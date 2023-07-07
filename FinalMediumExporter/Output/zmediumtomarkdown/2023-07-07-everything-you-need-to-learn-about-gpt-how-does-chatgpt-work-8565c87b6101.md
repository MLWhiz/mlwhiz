---
title: "Everything you need to learn about GPT — How Does ChatGPT Work?"
author: "Rahul Agarwal"
date: 2023-07-07T10:10:08.195+0100
last_modified_at: 2023-07-07T10:10:08.195+0100
categories: "MLWhiz"
tags: ["data-science","artificial-intelligence","chatgpt","machine-learning","programming"]
description: "Part I: GPT Series"
image:
  path: /assets/8565c87b6101/0*i0y2NlEz3IzIaQTA
---



![Photo by [Mariia Shalabaieva](https://unsplash.com/ja/@maria_shalabaieva?utm_source=medium&utm_medium=referral) on Unsplash](assets/8565c87b6101/0*i0y2NlEz3IzIaQTA)

Photo by [Mariia Shalabaieva](https://unsplash.com/ja/@maria_shalabaieva?utm_source=medium&utm_medium=referral) on Unsplash
#### GPT Series
### Everything you need to learn about GPT — How Does ChatGPT Work?
#### Part I: GPT Series

ChatGPT is what everyone is talking about nowadays\. Would it take all the jobs? Or would it result in misinformation on the web? There are just so many posts and articles that fill my inbox daily when it comes to GPTs\. Add to that so many versions of GPTs and tools to use these GPTs; it is getting increasingly frustrating to keep track of everything in the GPT Landscape\.

In this series of several posts about GPT, I intend to accumulate all the knowledge I have acquired over the months and arrange it logically into readable chunks for my readers\. I have planned a few posts, and the first post, i\.e\., this one, will discuss the various advancements in the NLP space that made ChatGPT possible and talk specifically about **_how chatGPT Works_** \. In the second post, I will take you through the **_ChatGPT API and how you could use the different ChatGPT endpoints_** \. In the third post, we will delve deeper into **_Prompt Engineering_** , where I will take you through the different tactics you can use to create usable prompts\. In the fourth Post, I will discuss **_Langchain_** and its various useful functions\. And I will keep on adding new posts over here\. So follow me up here on [**Medium**](https://mlwhiz.medium.com/) to be notified about these\.
### Evolution of ChatGPT


![Author Image: How ChatGPT Came to be](assets/8565c87b6101/1*huWymPGbq9WCV5utpoPIFg.png)

Author Image: How ChatGPT Came to be

As with everything, ChatGPT didn’t come into existence without a cumulative effort\. It had the shoulders of giants to climb upon\. ChatGPT Website says, _“ChatGPT is a sibling model to InstructGPT, which is trained to follow an instruction in a prompt and provide a detailed response\.”_ While InstructGPT is itself based on GPT3, which in turn is based on the transformer architecture\. So, if we try to look at the chronology, we will need to go through this series of papers to understand how the whole thing evolved to existence\. You should go through all these papers if you want to, but I have tried summarizing each below:
#### [**1\. Attention Is All You Need:**](https://arxiv.org/pdf/1706.03762.pdf)

In the past, the LSTM and GRU architecture\(as explained in my [post](https://towardsdatascience.com/nlp-learning-series-part-3-attention-cnn-and-what-not-for-text-classification-4313930ed566) on NLP\) and the attention mechanism used to be the State of Art Approach for Language modeling problems \(put very simply, predict the next word\) and Translation systems\. But, the main problem with these architectures is that they are recurrent, and the runtime increases as the sequence length increases\. These architectures take a sentence and sequentially process each word; hence, the whole runtime increases with the increase in sentence length\. Transformer, a model architecture first explained in the paper Attention is all you need, lets go of this recurrence and instead relies entirely on **_an attention mechanism to draw global dependencies between input and output_** \. Here are a few of my posts that delve deep into explaining Transformers\. For now, you can think of them as the building blocks of all the other GPTs that would come after it\.


[![](https://miro.medium.com/v2/resize:fit:1200/1*ug-MlPlFd6fhVrsdj1rDjw.jpeg)](https://towardsdatascience.com/understanding-transformers-the-data-science-way-e4670a4ee076)



[![](https://miro.medium.com/v2/resize:fit:1200/1*Sfu0EF2g-LdoAsmP4t_Q-Q.jpeg)](https://towardsdatascience.com/understanding-transformers-the-programming-way-f8ed22d112b2)

#### [2\. Improving Language Understanding by Generative Pre\-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

This paper which introduced GPT\-1, proposes pre\-training a model on a large corpus of unlabeled text using a generative model, **_followed by fine\-tuning on each specific task_** \. While BERT used Transformer architecture as both encoder and decoder, GPT architecture uses only multi\-layer Transformer decoder layers to create its network\. The training consists of two stages\.
1. **_Unsupervised pre\-training_** : This stage trains a network on unsupervised data to predict the next word given a context window of k words\.
2. **Supervised fine\-tuning:** This stage finetunes the network trained in stage 1 using a supervised dataset by adding a new layer on top of the stage 1 network and trying to minimize a weighted loss of classification loss and LM loss\.

#### [3\. **Language Models are Unsupervised Multitask Learners**](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)


![Examples of naturally occurring demonstrations of English to French and French to English translation found throughout the WebText training set\. Source: [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)](assets/8565c87b6101/1*HdscdbNQWn4aEpNuPG0E4Q.png)

Examples of naturally occurring demonstrations of English to French and French to English translation found throughout the WebText training set\. Source: [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

In this paper, the authors introduce the concept of unsupervised multitask learning, where a single language model is trained on a wide range of language tasks simultaneously\. The paper did two things:
- While the model's architecture remains similar to that of GPT\-1 with few changes in normalization layers, this was a very big model with 1\.5 Billion Parameters\.
- This model was not trained on the randomly chosen unsupervised corpus like Common Crawl but rather a dataset of millions of scraped webpages that emphasized document quality called WebText\. This allowed the model to get trained on more intelligible text that correlated more with the tasks the model was expected to solve\.

#### [4\. **Language Models are Few\-Shot Learners**](https://arxiv.org/pdf/2005.14165.pdf)

In this paper, the researchers introduce GPT\-3, an autoregressive language model with 175 billion parameters, and evaluate its performance in the few\-shot setting without fine\-tuning\. Particularly,
- The model is similar to GPT\-2 but with some modifications in the attention patterns of the transformer layers\.
- For training data, this model used a filtered version of CommonCrawl \+ Webtext they used for chatGPT \+ two internet\-based books corpora\.



![Source: [Language Models are Few\-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf)](assets/8565c87b6101/1*joMVvBgn0K7pqluidXbPhg.png)

Source: [Language Models are Few\-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf)
- This paper introduced the term “ **Few\-Shot Learning,** ” or at least I heard it for the first time here\. **_Few Shot_** is the term to refer to the setting where the model is given a few demonstrations of the task at inference time as conditioning but where no weight updates are allowed\. This paper evaluated its big pretrained language model with the few\-shot learning approach\. At evaluation time, the model is given examples of tasks and then asked to finish the task\. In a way, this was the start of moving from fine\-tuning to prompt engineering\.

#### [5\. Deep Reinforcement Learning from Human Preferences](https://arxiv.org/pdf/1706.03741.pdf)

This paper talks about how the reward function of certain tasks is hard to construct and introduces a method for training reinforcement learning agents using human feedback in the form of preferences\. The goal is to improve the performance and safety of AI systems by incorporating human values into the learning process\. The paper presents a framework called Reinforcement Learning from Human Feedback \(RLHF\), which utilizes a reward model generated from human preferences to guide the agent’s learning\. The process involves collecting pairwise comparisons from human evaluators to rank different agent behaviors\. This ranking is used to construct a reward model that guides the agent toward behaviors humans prefer\. The reinforcement model policy is then trained using the reward model\.
#### [6\. Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155.pdf)


![Source: [https://openai\.com/research/instruction\-following](https://openai.com/research/instruction-following)](assets/8565c87b6101/1*27sFm2tE81bqeb-nXfNOsw.png)

Source: [https://openai\.com/research/instruction\-following](https://openai.com/research/instruction-following)

The researchers at OpenAI trained this 80B parameter model using Reinforcement Learning from Human Feedback \(RLHF\) \. The researchers:

**Step 1** \. The authors first trained a baseline model, which involved taking a pre\-trained language model and fine\-tuning it using a small set of data carefully curated by labelers\. This fine\-tuning process aims to train a supervised policy known as the SFT model, which generates outputs based on specific prompts\.

**Step 2** \. To generate a reward model \(RM\), labelers are tasked with voting on many outputs generated by the SFT model\. This voting process creates a dataset of comparison data\. A new reward model is trained on this dataset, using the comparison data to guide be used in RLHF Step\. As we can understand, the voting process is much easier than manually generating the prompts and responses for Step 1 and hence is more scalable\.

**Step 3** \. In this step, the researchers used the SFT model as the baseline to initialize the PPO Model first and then used the rewards from the Reward model to change the weights of the PPO Model\. Simply put, they sent a prompt to the model, observed the reward from the reward model, and changed model weight parameters to maximize the reward\. In their objective, they also have certain terms that ensure that the SFT Model doesn’t diverge too much from the PPO Model for stability\.

**_Steps 2 and 3_** are then iterated continuously to create new, improved models\.


> **_And that is how ChatGPT came to be\._** 




ChatGPT, when it launched, was based on the GPT3\.5 Model, a model trained on GPT3 data along with the Open API data users interacted with\. ChatGPT is now based on GPT\-4 architecture as the baseline language model having a massive 170 trillion Parameters with RLHF used to train the PPO policy\. Here is the [GPT\-4 Technical Report](https://arxiv.org/pdf/2303.08774.pdf) , which specifies the various benchmarks this remarkable model has broken and how it compares with the past models\.


![Source: [GPT\-4 Technical Report](https://arxiv.org/pdf/2303.08774.pdf)](assets/8565c87b6101/1*tzBNEGV3HYL_Q-8JKXyAGw.png)

Source: [GPT\-4 Technical Report](https://arxiv.org/pdf/2303.08774.pdf)
### Conclusion

So, we were learning something each step of the way\. The Attention paper taught us about transformers, the GPT\-1 Paper about fine\-tuning on different tasks, the GPT\-2 Paper about using targeted datasets, the GPT\-3 Paper about few\-shot learning, the RLHF Paper about modeling rewards for RL using Human feedback, and the instructGPT Paper about using RLHF with GPT, which correspondingly came to be known as ChatGPT\.

In this blog post, I aimed to provide an overview of the evolutionary journey that led to the development of the ChatGPT model\. By examining various pivotal research papers, we have traced the path that brought us to the model's current state\. And to see that the research is still going at a breakneck pace and we see new things every day is nothing short of extraordinary\.

I will continue providing more info on these GPT models as we go through the GPT Series\. Let me know what you think about them\. Also, follow me up on [**Medium**](https://mlwhiz.medium.com/) or Subscribe to my [**blog**](https://mlwhiz.ck.page/a9b8bda70c) \. Optionally, you may also [**sign up**](https://medium.com/@mlwhiz/membership) for a Medium membership to get full access to every story on Medium\.



_[Post](https://mlwhiz.medium.com/everything-you-need-to-learn-about-gpt-how-does-chatgpt-work-8565c87b6101) converted from Medium by [ZMediumToMarkdown](https://github.com/ZhgChgLi/ZMediumToMarkdown)._
