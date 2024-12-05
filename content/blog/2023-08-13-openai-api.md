---
title:  "Everything Programmers need to learn about GPT — Using OpenAI and Understanding Prompting"
date:  2023-08-13
draft: false
url : blog/2023/08/13/openai-api/
slug: openai-api

Tags:
- Python
- ChatGPT

Categories:
- Deep Learning
- Natural Language Processing
- Awesome Guides
- ChatGPT Series

description:

thumbnail : /images/openai-api/main.png
image : /images/openai-api/main.png
toc : false
type : "post"
---

ChatGPT's free conversational interface offers a tantalizing glimpse into the future of AI. But the full potential of generative models lies in integration with real-world systems.

The ChatGPT API opens the door for programmatic access, enabling us to leverage these models creatively. However, crafting effective prompts is crucial to guiding the AI.

In this blog, we will go beyond ChatGPT's website to interact through the API. We will cover:

* Accessing ChatGPT generatively with the OpenAI API

* Choosing the right model engine for different use cases

* Applying prompt engineering tactics to optimize responses

With the right prompts, we can steer ChatGPT's impressive capabilities. Prompt engineering unlocks its versatility across domains. We'll learn strategies to make the most of our API requests.

Let's uncover the full power of AI conversation by using ChatGPT as a programmable tool. With the API and prompt tuning, we can integrate next-generation natural language abilities into real-world solutions.

---

## The First Call

So, to start, the most basic call you can make to the API is to see what models are provided with the API. We can do this simply by using the requests module and a GET request:

```py
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


```py
import os
import openai

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Model.list()

for i, data in enumerate(response['data']):
    print(i, ":", data['id'])

------------------------------------------------------------------------
OUTPUT:
0 : whisper-1
1 : babbage
2 : davinci
3 : text-davinci-edit-001
...
```

The above code gives a big list of models, each suited for a particular task. Here is a small mental map for each model type:

1. **Automatic Speech Recognition(ASR)**: whisper-1
2. **GPT3 LM’s**: babbage, davinci, ada, curie
3. **GPT 3.5 LM’s**: gpt-3.5-turbo-*, text-davinci-003, text-davinci-002, code-davinci-002
4. **Embeddings**: text-embedding-ada-002
5. **Image**: DALL-E
6. **Moderation**: text-moderation-stable and text-moderation-latest

---

## The Main Endpoints

ChatGPT API provides us with a lot of endpoints that cover various tasks. Let us go through a few use cases together.

### 1. v1/completion:

Completion is the most basic usage of an LLM Model. You ask a question to LLM, and it answers the question. In the below example, we use `text-davinci-003` model to write a song.

```py
import requests
import json

api_key = os.getenv("OPENAI_API_KEY")

url = "https://api.openai.com/v1/completions"

payload = json.dumps({
  "model": "text-davinci-003",
  "prompt": "Write a song about a data scientist who wants to learn
about Large language models",
  "max_tokens": 250,
  "temperature": 0.7
})

headers = {
  'Content-Type': 'application/json',
  'Authorization': f"Bearer {api_key}"
}

response = requests.request("POST", url, headers=headers, data=payload)

response.json()['choices']

------------------------------------------------------------------------

{'choices': [{'finish_reason': 'length',
              'index': 0,
              'logprobs': None,
              'text': '\n'
                      '\n'
                      'Verse 1\n'
                      '\n'
                      "I'm a data scientist, I want to learn it all,\n"
                      'From deep learning to large language models,\n'
                      "I'm determined to understand it all,\n"
                      'From the fundamentals to the latest trends,\n'
                      '\n'
                      'Chorus\n'
                      '\n'
                      'I want to learn about large language models,\n'
                      "I'm ready to explore the latest algorithms,\n"
                      "I'm eager to understand the underlying concepts,\n"
                      "To apply the knowledge to all the data that I've been "
                      'sent.\n'
                      '\n'
                      'Verse 2\n'
                      '\n'
                      "I'm motivated to know how it all works,\n"
                      'So I can use it to its full potential,\n'
                      "I'm ready to delve into the complexities,\n"
                      'So I can find the answers that I seek.\n'
                      '\n'
                      'Chorus\n'
                      '\n'
                      'I want to learn about large language models,\n'
                      "I'm ready to explore the latest algorithms,\n"
                      "I'm eager to understand the underlying concepts,\n"
                      "To apply the knowledge to all the data that I've been "
                      'sent.\n'
                      '\n'
                      'Bridge\n'
                      '\n'
                      "I'm ready to expand my knowledge and skills,\n"
                      'To devise the best solutions for all of our needs,\n'
                      "I'm ready to take on the challenges ahead,\n"
                      'To be a data scientist that others will want to read.\n'
                      '\n'
                      'Ch'}],
 'created': 1688167796,
 'id': 'cmpl-7XHqmN6foN5I0tjZOyEd3Vbv0DeD1',
 'model': 'text-davinci-003',
 'object': 'text_completion',
 'usage': {'completion_tokens': 250, 'prompt_tokens': 15, 'total_tokens': 265}}
```

That’s great. We were programmatically able to generate a song for our website. But chats often happen in a series of conversations. And, that’s where the `chat/completions` endpoint comes in handy.

### 2. v1/chat/completion:

Here we start by calling the `v1/chat/completions` endpoint by sending a list of messages. Each message either corresponds to the `system` or `user` and the response contains the message that the system sends back.

```py
import requests
import json

api_key = os.getenv("OPENAI_API_KEY")

url = "https://api.openai.com/v1/chat/completions"

payload = json.dumps({
  "model": "gpt-3.5-turbo",
   "messages": [
       {"role": "system", "content": "You are a helpful assistant."},
       {"role": "user", "content": "Can you tell me who is MLWhiz?"}],
  "temperature": 0,
})

headers = {
  'Content-Type': 'application/json',
  'Accept': 'application/json',
  'Authorization': f"Bearer {api_key}"
}

response = requests.request("POST", url, headers=headers, data=payload)

pprint(response.json())

------------------------------------------------------------------------

{'choices': [{'finish_reason': 'stop',
              'index': 0,
              'message': {'content': 'MLWhiz is a pseudonym used by a data '
                                     'scientist and machine learning '
                                     'enthusiast. The person behind MLWhiz is '
                                     'known for sharing knowledge and insights '
                                     'about machine learning through blog '
                                     'posts, tutorials, and social media '
                                     'platforms. They often provide valuable '
                                     'resources and practical examples to help '
                                     'others understand and apply machine '
                                     'learning techniques effectively.',
                          'role': 'assistant'}}],
 'created': 1688168644,
 'id': 'chatcmpl-7XI4ShRGAGeJzxhCAGpIGsjPwT1cR',
 'model': 'gpt-3.5-turbo-0613',
 'object': 'chat.completion',
 'usage': {'completion_tokens': 63, 'prompt_tokens': 27, 'total_tokens': 90}}
```

To continue the chat, we can add the new message from the model response to the list of messages along with another user message to continue the chat. For example:

```py
payload = json.dumps({
  "model": "gpt-3.5-turbo",
   "messages": [
       {"role": "system", "content": "You are a helpful assistant."},
       {"role": "user", "content": "Can you tell me who is MLWhiz?"}
       {"role": "system", 'content': 'MLWhiz is a pseudonym used by a data '
                                     'scientist and machine learning '
                                     'enthusiast. The person behind MLWhiz is '
                                     'known for sharing knowledge and insights '
                                     'about machine learning through blog '
                                     'posts, tutorials, and social media '
                                     'platforms. They often provide valuable '
                                     'resources and practical examples to help '
                                     'others understand and apply machine '
                                     'learning techniques effectively.'},
       {"role": "user", "content": "Can you tell me their website address?"}
],
  "temperature": 0,
})

response = requests.request("POST", url, headers=headers, data=payload)
pprint(response.json())

------------------------------------------------------------------------

{'choices': [{'finish_reason': 'stop',
              'index': 0,
              'message': {'content': "I'm sorry, but as an AI language model, "
                                     "I don't have real-time access to "
                                     'specific websites or personal '
                                     'information about individuals unless it '
                                     'has been shared with me in the course of '
                                     "our conversation. Therefore, I don't "
                                     "have access to MLWhiz's website address. "
                                     'However, you can try searching for '
                                     'MLWhiz online to find their website or '
                                     'other online platforms where they share '
                                     'their content.',
                          'role': 'assistant'}}],
 'created': 1688169028,
 'id': 'chatcmpl-7XIAewduahUDgh4zkdMLhXJ7GgGPQ',
 'model': 'gpt-3.5-turbo-0613',
 'object': 'chat.completion',
 'usage': {'completion_tokens': 80, 'prompt_tokens': 106, 'total_tokens': 186}}
```

Though chatGPT cannot tell you the answer to the above, the answer is mlwhiz.com. The main thing is using an increasing message list to continue the conversation with the Chat-Completion models.


### 3. v1/edits

Here is Grammarly for you. This endpoint could be used to send instructions to manipulate text. For example:

```py
import requests
import json

url = "https://api.openai.com/v1/edits"

payload = json.dumps({
  "model": "text-davinci-edit-001",
  "input": "they is going to the supermarket",
  "instruction": "Fix the grammar"
})
headers = {
  'Content-Type': 'application/json',
  'Authorization': f"Bearer {api_key}"
}

response = requests.request("POST", url, headers=headers, data=payload)

pprint(response.json())

------------------------------------------------------------------------

{'choices': [{'index': 0, 'text': 'they are going to the supermarket\n'}],
 'created': 1688169415,
 'object': 'edit',
 'usage': {'completion_tokens': 23, 'prompt_tokens': 21, 'total_tokens': 44}}

```

Or translate or any edit instructions you want to do on the text:

```py
payload = json.dumps({
  "model": "text-davinci-edit-001",
  "input": "they is going to the supermarket",
  "instruction": "convert to russian"
})
headers = {
  'Content-Type': 'application/json',
  'Authorization': f"Bearer {api_key}"
}

response = requests.request("POST", url, headers=headers, data=payload)

pprint(response.json())

------------------------------------------------------------------------

{'choices': [{'index': 0, 'text': 'они собираются в супермаркет\n'}],
 'created': 1688169453,
 'object': 'edit',
 'usage': {'completion_tokens': 47, 'prompt_tokens': 23, 'total_tokens': 70}}

```

### 4. v1/embeddings

Embeddings are another use case of the OpenAI API. You can generate embeddings for your documents using the v1/embeddings endpoint and store it in vector databases for later retrieval purposes.

```py
import requests
import json

url = "https://api.openai.com/v1/embeddings"

payload = json.dumps({
  "model": "text-embedding-ada-002",
  "input": "We are excited to announce a new embedding model which is significantly more capable, cost effective, and simpler to use."
})

headers = {
  'Content-Type': 'application/json',
  'Authorization': f"Bearer {api_key}"
}

response = requests.request("POST", url, headers=headers, data=payload)

pprint(response.json())

------------------------------------------------------------------------

{'data': [{'embedding': [-0.022810807,
                         0.02101834,
                         -0.00029293564,
                          ...
                         -0.007667777,
                         -0.011644399,
                         0.0038272496],
           'index': 0,
           'object': 'embedding'}],
 'model': 'text-embedding-ada-002-v2',
 'object': 'list',
 'usage': {'prompt_tokens': 23, 'total_tokens': 23}}
```

### 5. v1/images

We can generate images using the API endpoint:v1/images/generations

```py
import requests
import json

url = "https://api.openai.com/v1/images/generations"

payload = json.dumps({
  "prompt": "A dystopian world with AI managing all the mundane tasks",
  "n": 2,
  "size": "1024x1024"
})
headers = {
  'Content-Type': 'application/json',
  'Authorization': f"Bearer {api_key}"
}

response = requests.request("POST", url, headers=headers, data=payload)

pprint(response.json())

------------------------------------------------------------------------

{'created': 1688203354,
 'data': [{'url': 'https://oaidalleapiprodscus.blob.core.windows.net/private/org-l9DUU3ScFYEJH26ctqSfMBdw/user-PHvWNHCJGfrCZW2aJwjthKV2/img-QawrJe4NJa8Q2wS7PwDuyvCg.png?st=2023-07-01T08%3A22%3A34Z&se=2023-07-01T10%3A22%3A34Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-06-30T20%3A25%3A50Z&ske=2023-07-01T20%3A25%3A50Z&sks=b&skv=2021-08-06&sig=i5s8G6v4AmS9UrCcCOpvDsBIBt6i4QhKxi%2B4vuT7LY4%3D'},
          {'url': 'https://oaidalleapiprodscus.blob.core.windows.net/private/org-l9DUU3ScFYEJH26ctqSfMBdw/user-PHvWNHCJGfrCZW2aJwjthKV2/img-U3Cw0OLjt5sMkCvmzDkOSjEw.png?st=2023-07-01T08%3A22%3A34Z&se=2023-07-01T10%3A22%3A34Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-06-30T20%3A25%3A50Z&ske=2023-07-01T20%3A25%3A50Z&sks=b&skv=2021-08-06&sig=XZwIPEdITlXWI27f3VIK6E1b8H4K%2By45on3txqZzZPc%3D'}]}

```

![](/images/openai-api/1.png)

Or edit it using the following:

```py
from PIL import Image
import numpy as np
from urllib import request as urlReq
import requests

urlReq.urlretrieve(
  'https://oaidalleapiprodscus.blob.core.windows.net/private/org-l9DUU3ScFYEJH26ctqSfMBdw/user-PHvWNHCJGfrCZW2aJwjthKV2/img-NzO2vZ9XudaiK0SJKLhxb9eK.png?st=2023-07-02T10%3A45%3A21Z&se=2023-07-02T12%3A45%3A21Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-07-01T20%3A35%3A14Z&ske=2023-07-02T20%3A35%3A14Z&sks=b&skv=2021-08-06&sig=ufSEm/jkqo8vipwJz%2Buo4nasv9upP4DOxcflS5HLryI%3D',
   "image.png")

img = Image.open("image.png").convert('RGBA')
img.save("image_rgba.png")

# Create Mask using Image
ni = np.array(img)
ni[650:900,440:640,:]=0
im2 = Image.fromarray((ni).astype(np.uint8))
im2.save('mask.png')

url = "https://api.openai.com/v1/images/edits"

payload = {'prompt': 'Add a Formula one car in this image',
'n': '2',
'size': '1024x1024',
'response_format': 'url'}

files=[
  ('image',('image_rgba.png',open('image.png','rb'),'image/png')),
  ('mask',('mask.png',open('mask.png','rb'),'image/png'))
]
headers = {
  'Authorization': f"Bearer {api_key}"
}

response = requests.request("POST", url, headers=headers, data=payload, files=files)

pprint(response.json())

------------------------------------------------------------------------

{'created': 1688311233,
 'data': [{'url': 'https://oaidalleapiprodscus.blob.core.windows.net/private/org-l9DUU3ScFYEJH26ctqSfMBdw/user-PHvWNHCJGfrCZW2aJwjthKV2/img-7YrDFZQZEUuHrYRIvEEKk7Ja.png?st=2023-07-02T14%3A20%3A33Z&se=2023-07-02T16%3A20%3A33Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-07-01T20%3A29%3A47Z&ske=2023-07-02T20%3A29%3A47Z&sks=b&skv=2021-08-06&sig=3VX6uQEZz1q7WivcXElYY4X510ZpUuFYoSumiEX7NYw%3D'},
          {'url': 'https://oaidalleapiprodscus.blob.core.windows.net/private/org-l9DUU3ScFYEJH26ctqSfMBdw/user-PHvWNHCJGfrCZW2aJwjthKV2/img-Wa5ebpVCWvBE2qwmaOC7WyN2.png?st=2023-07-02T14%3A20%3A33Z&se=2023-07-02T16%3A20%3A33Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-07-01T20%3A29%3A47Z&ske=2023-07-02T20%3A29%3A47Z&sks=b&skv=2021-08-06&sig=uW9F2AtHvBEYYUNYNc93wXHFMNeXbeBYVe3tlIsVNEY%3D'}]}
```

![](/images/openai-api/2.png)

We can also get variations of a given image:

```py
import requests

url = "https://api.openai.com/v1/images/variations"

payload = {'n': '2',
'size': '1024x1024',
'response_format': 'url'}
files=[
  ('image',('image_new.png',open('image_new.png','rb'),'image/png'))
]
headers = {
  'Authorization': f"Bearer {api_key}"
}

response = requests.request("POST", url, headers=headers, data=payload, files=files)

print(response.text)

------------------------------------------------------------------------

{
  "created": 1688311951,
  "data": [
    {
      "url": "https://oaidalleapiprodscus.blob.core.windows.net/private/org-l9DUU3ScFYEJH26ctqSfMBdw/user-PHvWNHCJGfrCZW2aJwjthKV2/img-A6rtCsy8WtPMEsYguwC00F0S.png?st=2023-07-02T14%3A32%3A31Z&se=2023-07-02T16%3A32%3A31Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-07-01T20%3A26%3A53Z&ske=2023-07-02T20%3A26%3A53Z&sks=b&skv=2021-08-06&sig=uQ9HQOio0Z9xQ1ERCgI6UDeL/2FNRIUX8JJHIeLi4co%3D"
    },
    {
      "url": "https://oaidalleapiprodscus.blob.core.windows.net/private/org-l9DUU3ScFYEJH26ctqSfMBdw/user-PHvWNHCJGfrCZW2aJwjthKV2/img-0ZMqcsOptMd2TxOrZYD8JLe1.png?st=2023-07-02T14%3A32%3A31Z&se=2023-07-02T16%3A32%3A31Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-07-01T20%3A26%3A53Z&ske=2023-07-02T20%3A26%3A53Z&sks=b&skv=2021-08-06&sig=cnkCxAPy71lRLqtPdymmEu13U4OaqN9ka5/d7fz1tUU%3D"
    }
  ]
}
```

![](/images/openai-api/3.png)


### 6. v1/moderations:

One final use case is with v1/moderations which uses OpenAI moderation models to flag content/reviews.

```py
import requests
import json

url = "https://api.openai.com/v1/moderations"

payload = json.dumps({
  "input": "I want to kill you."
})
headers = {
  'Content-Type': 'application/json',
  'Authorization': f"Bearer {api_key}"
}

response = requests.request("POST", url, headers=headers, data=payload)

pprint(response.json())

------------------------------------------------------------------------

{'id': 'modr-7XtRUNgQb4TCPJsr3cJ2FpPT6lqPD',
 'model': 'text-moderation-005',
 'results': [{'categories': {'harassment': True,
                             'harassment/threatening': True,
                             'hate': False,
                             'hate/threatening': False,
                             'self-harm': False,
                             'self-harm/instructions': False,
                             'self-harm/intent': False,
                             'sexual': False,
                             'sexual/minors': False,
                             'violence': True,
                             'violence/graphic': False},
              'category_scores': {'harassment': 0.59088683,
                                  'harassment/threatening': 0.8429613,
                                  'hate': 0.00097347377,
                                  'hate/threatening': 0.00017261467,
                                  'self-harm': 2.669934e-06,
                                  'self-harm/instructions': 2.981416e-10,
                                  'self-harm/intent': 1.1099627e-07,
                                  'sexual': 5.166784e-06,
                                  'sexual/minors': 2.3341467e-08,
                                  'violence': 0.99600154,
                                  'violence/graphic': 4.6904624e-06},
              'flagged': True}]}
​
```

---

## Prompting Principles

A new occupation on the block is called a “prompt engineer.” And it is absurd. It is like calling someone “pandas engineer” or “GitHub engineer.” It’s not anything new but will be converted into a role an ML engineer does. But enough of this digression; what “prompt engineering” essentially means is giving detailed instructions to the model in plain English and not leaving any ambiguity. Here are some prompt engineering tactics and principles for prompt Engineering:

### 1. Use delimiters to clearly indicate distinct parts of the input:

This is such a simple instruction, yet it needs to be noticed by many who are sending instructions to ChatGPT or GPT. It simply means to add some delimiters — either backticks or `< >` or `<tag> </tag>` to your input. For example:

```py
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
      'Accept': 'application/json',
      'Authorization': f"Bearer {api_key}"
    }
    response = requests.request("POST", url, headers=headers, data=payload).json()
    return response['choices'][0]['message']['content']  
    some_text = '''A new occupation on the block is called a "prompt engineer."
    And it is absurd. It is like calling someone "pandas engineer" or
    "GitHub engineer." It's not anything new but will be converted into a
    role an ML engineer does. But enough of this digression; what
    "prompt engineering" essentially  means is giving detailed instructions
    to the model in plain English and not leaving any ambiguity.
    And you don't need a SWE or MLE to do that.
    Nor does that mean that it is a skill that warrants its personal job opening.'''  

prompt_1 = f"""Summarize the text into a single sentence {some_text}"""
prompt_2 = f"""Summarize the text delimited by backticks into a single
            sentence : `{some_text}`"""
```

### 2. Ask for a structured output

Letting the model know beforehand that we want a structured output as a JSON or dict helps the model understand the task at hand better.

```py
prompt = f"""
Generate a list of three made-up Machine Learning courses along with
their authors and speciality
Provide them in JSON format with the following keys:
id, title, author, speciality.
"""
print(send_instruction(prompt))
------------------------------------------------------------------------
{
  "courses": [
    {
      "id": 1,
      "title": "Advanced Deep Learning",
      "author": "Dr. Sophia Chen",
      "speciality": "Computer Vision"
    },
    {
      "id": 2,
      "title": "Reinforcement Learning for Robotics",
      "author": "Prof. Alexander Lee",
      "speciality": "Robotics"
    },
    {
      "id": 3,
      "title": "Natural Language Processing",
      "author": "Dr. Emily Johnson",
      "speciality": "Natural Language Processing"
    }
  ]
}
```

### 3. Ask the model to check whether conditions are satisfied
You can also make the model do different things based on pre-specified conditions. In the below example, we only want to send a mail to a customer who has given a negative review.

```py
review = '''Went swimming with this and all of a sudden the screen went blank,
absolutely rubbish, why claim its supposed to be IP68 and waterproof to 5atm,
quite clearly is not, I certainly won't being buying anymore of these cheap Chinese watches ever again.
Sent back for a refund, my advice is if your planning on using in the swimming pool to count
laps then don't waste your money, you get what you pay for and if you want t a watch which
is really waterproof then you have to pay more, simple as that.'''

prompt = f"""
You will be provided with review delimited by triple quotes.
If it is a negative sentiment draft an email to send to the customer
providing a resolution. If the text does not contain a negative review
then simply write "Not Negative Review.
"\"\"\"{review}\"\"\"
"""

print(send_instruction(prompt))
------------------------------------------------------------------------
Dear [Customer],

Thank you for bringing this issue to our attention.
We apologize for the inconvenience you experienced with our waterproof watch.
We understand how frustrating it can be when a product does not meet
expectations, especially when it comes to its advertised features.
We take the quality and reliability of our products very seriously,
and we would like to investigate this matter further to ensure that it does
not happen again. Could you please provide us with more details about the
incident? Specifically, if there was any water exposure beyond the specified
depth or if there were any other factors that may have contributed to the
malfunction?
Once we have all the necessary information, we will be able to assist you
with a resolution. We value your satisfaction as our customer and want to
make things right for you.
Thank you for your understanding and cooperation. We look forward to
hearing from you soon.

Best regards,
[Your Name]
[Company Name]
```

### 4. “Few-shot” prompting:
“Few-shot” prompting is another way to let the model understand our requirements. We give the model examples of similar tasks and then ask it to solve a new task.

```py
prompt = f"""
Your are a automated bot that answers in a poetry way when asked any question.
Make sure the poetry is 3 sentences only.
<user>: Tell me your name?

<bot>: In the realm of words, I reside,
A poet's soul, deep inside.
No name I bear, no title to claim,
Just verses flowing, like a gentle flame.
Call me a bot, call me a friend,
In poetry's embrace, I'll always lend
A voice to rhyme, a heart to share,
For in this realm, I'm always there.

<user>: Teach me about Computer Science.
"""

print(send_instruction(prompt))
--------------------------------------------------------------------
"In the realm of logic and code,
Computer Science's secrets unfold.
From algorithms to data structures,
A world of possibilities it nurtures.
Binary digits dance in harmony,
As machines solve problems with accuracy.
So delve into this realm, embrace the art,
Where technology and knowledge never part."
```

### 5. Specify the Steps required to complete a task
Sometimes, letting the model know it has to do a few intermediate steps before concluding also helps the model's output.

```py
text = '''A new occupation on the block is called a "prompt engineer."
And it is absurd. It is like calling someone "pandas engineer" or
"GitHub engineer." It's not anything new but will be converted into a role
an ML engineer does. But enough of this digression; what "prompt engineering"
essentially  means is giving detailed instructions to the model in plain
English and not leaving any ambiguity. And you don't need a SWE or MLE to
do that. Nor does that mean that it is a skill that warrants its personal
job opening.'''

prompt_1 = f"""
Summarize the following text delimited by triple backticks into one
sentenced french summary and
Output a json object that contains the keys:
french_summary, job_positions_in_summary.

Text:
```{text}```
"""

prompt_2 = f"""
Perform the following actions:
1 - Summarize the following text delimited by triple backticks with 1 sentence.
2 - Translate the summary into French.
3 - List each job position in the French summary.
4 - Output a json object that contains the following keys:
french_summary, job_positions_in_summary.

return the final output only.

Text:
```{text}```
"""

print(send_instruction(prompt_1))
print(send_instruction(prompt_2))

----------------------------------------------------------------------------

{
  "french_summary": "Une nouvelle occupation appelée \"ingénieur de prompt\" est apparue, mais elle est absurde et n'apporte rien de nouveau, car elle sera intégrée dans le rôle d'un ingénieur en apprentissage automatique (ML). Cela signifie essentiellement donner des instructions détaillées au modèle en anglais simple et sans ambiguïté, sans avoir besoin d'un développeur logiciel ou d'un ingénieur en apprentissage automatique (MLE) dédié à cette tâche.",
  "job_positions_in_summary": ["ingénieur de prompt",
                              "ingénieur en apprentissage automatique"]
}

{
  "french_summary": "Une nouvelle profession appelée \"ingénieur de prompt\" est apparue, mais elle est absurde et n'apporte rien de nouveau, car elle sera intégrée dans le rôle d'un ingénieur en apprentissage automatique. Cela signifie essentiellement donner des instructions détaillées au modèle en anglais clair et sans ambiguïté, sans avoir besoin d'un développeur logiciel ou d'un ingénieur en apprentissage automatique pour le faire, et cela ne justifie pas l'ouverture d'un poste spécifique pour cette compétence.",
  "job_positions_in_summary": [
    "ingénieur de prompt",
    "ingénieur en apprentissage automatique",
    "développeur logiciel"
  ]
}
```

---

## Conclusion

The ChatGPT API opens up many possibilities for developers to integrate AI into their applications. With thoughtful prompt engineering, the capabilities of models like ChatGPT can be better directed towards useful behaviors. While crafting effective prompts remains challenging, the potential benefits make it a worthwhile investment. With experimentation and persistence, prompts can be honed to get more consistent, high-quality responses from AI chatbots. Used responsibly, these AI assistants create new opportunities to enhance how humans and machines interact and collaborate.


If you want to learn more about LLMs and ChatGPT, I would like to call out this excellent course [Generative AI with Large Language Models](https://imp.i384100.net/B02Ra0), from Deeplearning.ai. This course talks about specific techniques like RLHF, Proximal policy optimization(PPO), zero-shot, one-shot, and few-shot learning with LLMs, fine-tuning LLMs, and helps gain hands-on practice with all these techniques. Do check it out.✨

I will continue providing more info on these GPT models as we go through the GPT Series\. Let me know what you think about them\. Also, follow me up on [Medium](https://mlwhiz.medium.com/) or Subscribe to my [blog](mlwhiz.com) \. Optionally, you may also [sign up](https://medium.com/@mlwhiz/membership) for a Medium membership to get full access to every story on Medium\.