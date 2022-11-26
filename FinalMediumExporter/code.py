#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import re
import argparse
import requests
from datetime import datetime
import os
from glob import glob
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--blog', help="BlogLink")
parser.add_argument('-o', '--output', help="Output Folder Name")
args = parser.parse_args()

blog = args.blog
output = args.output

os.system(f"rm -rf Output")

os.system(f"ZMediumToMarkdown -p {blog}")
out_path = glob("Output/zmediumtomarkdown/*.md")[0]
images_list = glob("Output/zmediumtomarkdown/assets/*/*")

f = open(f"{out_path}","r")
contents =f.read()
out_path = glob("Output/zmediumtomarkdown/*.md")[0]
images_list = glob("Output/zmediumtomarkdown/assets/*/*")

f= open(f"{out_path}","r")
# In[3]:
contents =f.read()

# replace image links

img_dir = "/Users/ragarwal/personal/web/mlwhiz/static/images"
post_dir = "/Users/ragarwal/personal/web/mlwhiz/content/blog"
os.system(f"mkdir {img_dir}/{output}")

ext=None

for i,img_name in enumerate(images_list):
    if i==0:
        os.system(f"cp {img_name} {img_dir}/{output}/main.{img_name.split('.')[-1]}")
        ext = img_name.split('.')[-1]
        print(f"cp {img_name} {img_dir}/{output}/main.{img_name.split('.')[-1]}")
        
    os.system(f"cp {img_name} {img_dir}/{output}")
    contents = contents.replace("/".join((img_name.split("/")[-3:])), 
                     f"/images/{output}/{img_name.split('/')[-1]}"
                    )
    print("/".join((img_name.split("/")[-3:])), 
                     f"/images/{output}/{img_name.split('/')[-1]}"
                    )
                    

title = contents.split("title:")[1].split("\n")[0].strip()
pattern = re.compile(r"---\n[\s\S]*\n---\n")
contents = contents.replace(pattern.findall(contents)[0],"")

date = datetime.today().strftime('%Y-%m-%d')
contents = contents[contents.find("####"):]

text = f'''
---
title:  {title}
date:  {date}
draft: false
url : blog/{date.split("-")[0]}/{date.split("-")[1]}/{date.split("-")[2]}/{output}/
slug: {output}

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

thumbnail : /images/{output}/main.{ext}
image : /images/{output}/main.{ext}
toc : false
type : "post"
---

'''

contents=text+contents
contents = contents[:contents.find("_Converted")]
contents = contents.replace("\n#### ", "\n&&&&&&&&&&&&&&&& ")
contents = contents.replace("\n### ", "\n\n---\n## ")
contents = contents.replace("\n&&&&&&&&&&&&&&&& ", "\n### ")
contents = contents.replace("[**blog**](https://mlwhiz.ck.page/a9b8bda70c)","[blog](https://mlwhiz.ck.page/a9b8bda70c)")
contents = contents.replace("[**Medium**](http://mlwhiz.medium.com/)","[Medium](http://mlwhiz.medium.com/)")
regex = r'!\[[^\]]*\]\((.*?)\s*("(?:.*[^"])")?\s*\)\n\n'
matches = re.finditer(regex, contents, re.MULTILINE)
for m in matches:
    str_new = m.group()
    enclosed = str_new.split("[")[1].split("]")[0]
    str_to_replace = str_new+enclosed+"\n"
    to_replace_with = str_new.strip()[:-1]+' "' +enclosed+'")\n'
    contents = contents.replace(str_to_replace,to_replace_with)

file1 = open(f"{post_dir}/{date}-{output}.md", "w")
file1.write(contents)

file1.close
