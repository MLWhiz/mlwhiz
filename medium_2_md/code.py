#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import re
import argparse
import requests
from datetime import datetime
import os
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--blog', help="BlogLink")
parser.add_argument('-o', '--output', help="Output Folder Name")
args = parser.parse_args()

os.system(f"mediumexporter {args.blog} > {args.output}.md")
f= open(f"{args.output}.md","r")
# In[3]:
contents =f.read()
#print(contents)
title = contents.split("\n")[1].replace("#","")

c = contents.split("\n")
newc = []
for i,cin in enumerate(c):
    print(i,cin)
    if '###' not in cin:
        continue
    else:
        contents = "\n".join(c[i+1:])
        break
# In[8]:

contents = contents.replace("\n## ", "\n---\n## ")

contents = contents.replace("[**Medium](https://medium.com/@rahul_agarwal?source=post_page---------------------------)**", "[Medium](https://medium.com/@rahul_agarwal?source=post_page---------------------------)")

contents = contents.replace("[**blog](http://eepurl.com/dbQnuXx?source=post_page---------------------------)**", "[blog](http://eepurl.com/dbQnuX?source=post_page---------------------------)")

contents = contents.replace("[**@mlwhiz](https://twitter.com/MLWhiz?source=post_page---------------------------)**", "[@mlwhiz](https://twitter.com/MLWhiz?source=post_page---------------------------)**")

contents = contents.replace("[**Medium](https://medium.com/@rahul_agarwal)**", "[Medium](https://medium.com/@rahul_agarwal)")
contents = contents.replace("[**blog](http://eepurl.com/dbQnuX)**", "[blog](http://eepurl.com/dbQnuXx)")

pattern = r'https:\/\/cdn\-images\-1.medium.com[\/\w\*\.\-]*'

urls = re.findall(pattern, contents)

import os

# remember the / at end

img_dir = "/home/rahul/projects/web/new_blog/mlwhiz/static/images"
post_dir = "/home/rahul/projects/web/new_blog/mlwhiz/content/blog"
os.system(f"mkdir {img_dir}/{args.output}")

# In[10]:

blg_source = requests.get(args.blog)
main = r'https:\/\/miro.medium.com\/max\/2000[\/\w\*\.\-]*'

try:
    main_img_url = re.findall(main, blg_source.text)[0]

    ext = main_img_url.split(".")[-1]
    if ext not in ["png",'jpg','jpeg', 'gif']:
        ext = 'png'
    f = open(f"{img_dir}/{args.output}"+"/"+'main'+'.png', 'wb')
    f.write(requests.get(main_img_url).content)
    f.close()
except:
    pass
for i, url in enumerate(urls):
    print(url)
    ext = url.split(".")[-1]
    if ext in ["png",'jpg','jpeg', 'gif']:

        fname = f"{img_dir}/{args.output}"+"/"+str(i)+'.png'
        f = open(f"{img_dir}/{args.output}"+"/"+str(i)+"."+ext, 'wb')
        f.write(requests.get(url).content)
        f.close()

namedict = {}
for i, url in enumerate(urls):
    ext = url.split(".")[-1]
    namedict[url]="/images/"+args.output+"/"+str(i)+'.'+ext

for k,v in namedict.items():
    contents = contents.replace(k, v)
    date = datetime.today().strftime('%Y-%m-%d')

text = f'''
---
title:  {title}
date:  {date}
draft: false
url : blog/{date.split("-")[0]}/{date.split("-")[1]}/{date.split("-")[2]}/{args.output}/
slug: {args.output}

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

thumbnail : /images/{args.output}/main.png
image : /images/{args.output}/main.png
toc : false
type : "post"
---

'''

contents=text+contents

#write a string to a file

file1 = open(f"{post_dir}/{date}-{args.output}.md", "w")
file1.write(contents)

file1.close

