
---
title:  "Everything Programmers need to learn about GPT — OpenAI API in Action"
date:  2023-08-13
draft: false
url : blog/2023/08/13/openai-api/
slug: openai-api

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

thumbnail : /images/openai-api/main.png
image : /images/openai-api/main.png
toc : false
type : "post"
---

### GPT Series

---
## Everything Programmers need to learn about GPT — OpenAI API in Action
### Part II: GPT Series

Unless you have been living under a rock, by now, you would have used the ChatGPT interface one way or another on their website\. The free version provides pretty good functionality, which is why it was one of the most talked about topics when it launched\. But the API version offers even more regarding development with OpenAI models\. And you will need to learn to use the APIs to integrate ChatGPT into your production systems\.

In this series of several posts about GPT, I intend to accumulate all the knowledge I have acquired over the months and arrange it logically into readable chunks for my readers\. In my last post, I discussed the various advancements in the NLP space that made ChatGPT possible and talked specifically about [**_how chatGPT Works_**](https://medium.com/the-algorithmic-minds/everything-you-need-to-learn-about-gpt-how-does-chatgpt-work-8565c87b6101) \. In this post, I will take you through the **_ChatGPT API and how you could use the different ChatGPT endpoints_** \. In the third post, we will delve deeper into **_Prompt Engineering_** , where I will take you through the different tactics you can use to create usable prompts\. In the fourth Post, I will discuss **_Langchain_** and its various useful functions\. And I will keep on adding new posts over here\. So follow me up here on [**Medium**](https://mlwhiz.medium.com/) to be notified about these\.

---
## The First Call

So, to start, the most basic call you can make to the API is to see what models are provided with the API\. We can do this simply by using the requests module and a GET request:
```python
import requests

url = "https://api.openai.com/v1/models"

payload = {}
headers = {
  'Authorization': 'Bearer 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'
}

response = requests.request("GET", url, headers=headers, data=payload)

for i, data in enumerate(response.json()['data']):
    print(i, ":", data['id'])

-----
OUTPUT:
0 : whisper-1
1 : babbage
2 : davinci
3 : text-davinci-edit-001
...
```

Or an equivalent request using the `openai` Python module:
```python
import os
import openai

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Model.list()

for i, data in enumerate(response['data']):
    print(i, ":"…
```



_[Post](https://medium.com/the-algorithmic-minds/everything-you-need-to-learn-about-gpt-openai-api-in-action-bf210908df1f) converted from Medium by [ZMediumToMarkdown](https://github.com/ZhgChgLi/ZMediumToMarkdown)._