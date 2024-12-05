---
title: "3 Great Additions for your Jupyter Notebooks"
date:  2019-06-28
draft: false
url : blog/2019/06/28/jupyter_extensions/
slug: jupyter_extensions
Category: Python
Keywords:
- Python

Categories:
- Data Science

Tags:
- Python
- Machine Learning
- Data Science
- Tools
- Productivity
description: I love Jupyter notebooks and the power they provide.They can be used to present findings as well as share code in the most effective manner which was not easy with the previous IDEs.Yet there is something still amiss.There are a few functionalities I aspire in my text editor which don’t come by default in Jupyter.But fret not. Just like everything in Python, Jupyter too has third-party extensions. This post is about some of the most useful extensions I found.
thumbnail : /images/extensions/nbext_snippets.gif
image : /images/extensions/nbext_snippets.gif
toc : false
type : post
---


<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/extensions/start.png""></center>
</div>

I love Jupyter notebooks and the power they provide.

They can be used to present findings as well as share code in the most effective manner which was not easy with the previous IDEs.

Yet there is something still amiss.

There are a few functionalities I aspire in my text editor which don’t come by default in Jupyter.

But fret not. Just like everything in Python, Jupyter too has third-party extensions.

***This post is about some of the most useful extensions I found.***

---

## 1. Collapsible Headings

The one extension, I like most is collapsible headings.

It makes the flow of the notebook easier to comprehend and also helps in creating presentable notebooks.

To get this one, install the `jupyter_contrib_nbextensions` package with this command on the terminal window:

    conda install -c conda-forge jupyter_contrib_nbextensions

Once the package is installed, we can start jupyter notebook using:

    jupyter notebook

Once you go to the home page of your jupyter notebook, you can see that a new tab for NBExtensions is created.

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/extensions/jupyter.png""></center>
</div>

And we can get a lot of extensions using this package.

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/extensions/nbext.png""></center>
</div>

This is how it looks:

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/extensions/nbextension_collapsible.gif""></center>
</div>

---

## 2. Automatic Imports

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/extensions/auto.jpeg""></center>
</div>

Automation is the future.

One thing that bugs me is that whenever I open a new Jupyter notebook in any of my data science projects, I need to copy paste a lot of libraries and default options for some of them.

To tell you about some of the usual imports I use:

* Pandas and numpy — In my view, Python must make these two as a default import.

* Seaborn, matplotlib, plotly_express

* change some pandas and seaborn default options.

Here is the script that I end up pasting over and over again.
```py
import pandas as pd
import numpy as np

import plotly_express as px
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

*# We dont Probably need the Gridlines. Do we? If yes comment this line*
sns.set(style="ticks")

# pandas defaults
pd.options.display.max_columns = 500
pd.options.display.max_rows = 500
```
***Is there a way I can automate this?***

Just go to the nbextensions tab and select the snippets extension.

You will need to make the following changes to the snippets.json file. You can find this file at `/miniconda3/envs/py36/share/jupyter/nbextensions/snippets` location. The py36 in this location here is my conda virtualenv. It took me some time to find this location for me. Yours might be different. Please note that you don’t have to change at the site-packages location.

```json
{
    "snippets" : [
        {
            "name" : "example",
            "code" : [
                "# This is an example snippet!",
                "# To create your own, add a new snippet block to the",
                "# snippets.json file in your jupyter nbextensions directory:",
                "# /nbextensions/snippets/snippets.json",
                "import this"
            ]
        },

        {
            "name" : "default",
            "code" : [
                "# This is A snippet for all data related tasks",
                "import pandas as pd"
                "import numpy as np"
                "import plotly_express as px"
                "import seaborn as sns"
                "import matplotlib.pyplot as plt"
                "%matplotlib inline"
                "# We dont Probably need the Gridlines. Do we? If yes comment this line"
                "sns.set(style='ticks')"
                "# pandas defaults"
                "pd.options.display.max_columns = 500"
                "pd.options.display.max_rows = 500"
            ]
        }
    ]
}
```

You can see this extension in action below.

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/extensions/nbext_snippets.gif""></center>
</div>

Pretty cool. Right? I also use this to create basic snippets for my deep learning notebooks and NLP based notebooks.

---

## 3. Execution Time

We have used `%time` as well as decorator based timer functions to measure time for our functions. You can also use this excellent extension to do that.

Plus it looks great.

Just select the ExecutionTime extension from the NBextensions list and you will have an execution result at the bottom of the cell after every cell execution as well as the time when the cell was executed.

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/extensions/time.png""></center>
</div>

---

## Other Extensions

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/extensions/others.jpeg""></center>
</div>

NBExtensions has a lot of extensions. Some other extensions from NBExtensions I like and you might want to look at:

* **Limit Output:** Ever had your notebook hang since you printed a lot of text in your notebook. This extension limits the number of characters that can be printed below a code cell

* **2to3Convertor:** Having problems with your old python2 notebooks. Tired of changing the print statements. This one is a good one.

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/extensions/demo_2to3.gif""></center>
</div>

* **Live Markdown Preview:** Some of us like writing our blogs using Markdown in a jupyter notebook. Sometimes it can be hectic as you make errors in writing. Now you can see Live-preview of the rendered output of markdown cells while editing their source.

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/extensions/markdownpreview.gif""></center>
</div>

---

## Conclusion

I love how there is a package for everything with Python. And that holds good with the Jupyter notebook too.

The `jupyter_contrib_nbextensions` package works great out of the box.

It has made my life a lot easier when it comes to checking execution times, scrolling through the notebook, and repetitive tasks.

There are many other extensions this package does provide. Do take a look at them and try to see which ones you find useful.

Also, if you want to learn more about Python 3, I would like to call out an excellent course on Learn [Intermediate level Python](https://imp.i384100.net/6yyWGV) from the University of Michigan. Do check it out.

I am going to be writing more of such posts in the future too. Let me know what you think about the series. Follow me up at [**Medium**](https://mlwhiz.medium.com/) or Subscribe to my [**blog**](mlwhiz.com).
