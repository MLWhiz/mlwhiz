
---
title:  "Everything you need to learn about GPT — What is Prompt Engineering?"
date:  2023-08-13
draft: false
url : blog/2023/08/13/prompt-engineering/
slug: prompt-engineering

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

thumbnail : /images/prompt-engineering/main.png
image : /images/prompt-engineering/main.png
toc : false
type : "post"
---

### GPT Series

---
## Everything you need to learn about GPT — What is Prompt Engineering?
### Part III: GPT Series

A new occupation on the block is called a “ _prompt engineer\.”_ And it is absurd\. It is like calling someone _“pandas engineer”_ or _“GitHub engineer\.”_ It’s not anything new but will be converted into a role an ML engineer does\. But enough of this digression; what “prompt engineering” essentially means is giving detailed instructions to the model in plain English and not leaving any ambiguity\. And you don’t need a SWE or MLE to do that\. Nor does that mean that it is a skill that warrants its personal job opening\.

In this series of several posts about GPT, I intend to accumulate all the knowledge I have acquired over the months and arrange it logically into readable chunks for my readers\. In my last post, I discussed the various advancements in the NLP space that made ChatGPT possible and talked specifically about [**_how chatGPT Works_**](https://medium.com/the-algorithmic-minds/everything-you-need-to-learn-about-gpt-how-does-chatgpt-work-8565c87b6101?source=your_stories_page-------------------------------------) \. In the second post, we reviewed the ChatGPT API and how to **_use the [different ChatGPT endpoints](https://medium.com/the-algorithmic-minds/everything-you-need-to-learn-about-gpt-openai-api-in-action-bf210908df1f?source=your_stories_page-------------------------------------)_** \. In this post, we will delve deeper into **_Prompt Engineering_** , where I will take you through the different tactics you can use to create usable prompts\. And I will keep on adding new posts over here\. So follow me up here on [**Medium**](https://mlwhiz.medium.com/) to be notified about these\.

---
## Prompting Principles:

Here are some prompt engineering tactics and principles from this free [Andrew Ng course on Prompt Engineering](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/) :
- **_Use delimiters to clearly indicate distinct parts of the input_** : _This is such a simple instruction, yet it needs to be noticed by many who are sending instructions to ChatGPT or GPT\. It simply means to add some delimiters — either ````` or `< >` or `<tag> </tag>` to your input\. For example:_

```python
import requests
import json

api_key = os.getenv("OPENAI_API_KEY")

def send_instruction(prompt, model="gpt-3.5-turbo", temperature=0):
    url = "https://api.openai.com/v1/chat/completions"
    payload = json.dumps({
      "model": model,
       "messages": [
           {"role": "system", "content": "You are a helpful assistant."}, 
           {"role": "user", "content": prompt}],
      "temperature": temperature,
    })
    headers = {
      'Content-Type': 'application/json',
      'Accept': 'application/json'…
```



_[Post](https://medium.com/the-algorithmic-minds/everything-you-need-to-learn-about-gpt-what-is-prompt-engineering-15ce58d3e202) converted from Medium by [ZMediumToMarkdown](https://github.com/ZhgChgLi/ZMediumToMarkdown)._