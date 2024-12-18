---
title: "Why Sublime Text for Data Science is Hotter than Jennifer Lawrence?"
date:  2019-03-31
draft: false
url : blog/2019/03/31/sublime_ds_post/
slug: sublime_ds_post
Category: data science, Sublime
Keywords:
- data science
- productivity tools
- sublime text
- excel
- tools

Categories:
- Data Science

Tags:
- Data Science
- Tools

description: For a practitioner in any field, they turn out as good as the tools they use. Data Scientists are no different. But sometimes we don't even know which tools we need and also if we need them. We are not able to fathom if there could be a more natural way to solve the problem we face. In this post, I will try to talk about the Sublime Text Editor in the context of Data Science.

thumbnail : /images/sublime_ds/sublime_tool.jpeg
image :  /images/sublime_ds/sublime_tool.jpeg
toc : false
type : post
---



Just Kidding, Nothing is hotter than Jennifer Lawrence. But as you are here, let's proceed.

For a practitioner in any field, they turn out as good as the tools they use. Data Scientists are no different. But sometimes we don't even know which tools we need and also if we need them. We are not able to fathom if there could be a more natural way to solve the problem we face. We could learn about Data Science using awesome MOOCs like [Machine Learning](https://imp.i384100.net/Z66Xe1) by Andrew Ng but no one teaches the spanky tools of the trade. This motivated me to write about the tools and skills that one is not taught in any course in my new series of short posts - **Tools For Data Science**. As it is rightly said:

> We shape our tools and afterward our tools shape us.

In this post, I will try to talk about the Sublime Text Editor in the context of Data Science.

Sublime Text is such a lifesaver, and we as data scientists don't even realize that we need it. We are generally so happy with our Jupyter Notebooks and R studio that we never try to use another editor.

So, let me try to sway you a little bit from your Jupyter notebooks into integrating another editor in your workflow. I will try to provide some use cases below. On that note, these use cases are not at all exhaustive and are here just to demonstrate the functionality and Sublime power.

---

## 1. Create A Dictionary/List or Whatever:

How many times does it happen that we want to ***make a list or dictionary for our Python code from a list we got in an email text?*** I bet numerous times. 

How do we do this? We haggle in Excel by loading that Text in Excel and then trying out concatenating operations. For those of us on a Mac, it is even more troublesome since Mac's Excel is not as good as windows(to put it mildly)

So, for example, if you had information about State Name and State Short Name and you had to create a dictionary for Python, you would end up doing something like this in Excel. Or maybe you will load the CSV in pandas and then play with it in Python itself.

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/sublime_ds/previous_excel-min.gif" ></center>
</div>

*Here is how you would do the same in Sublime.* And see just how wonderful it looks. We ended up getting the Dictionary in one single line. It took me around 27 seconds to do. 

I still remember the first time I saw one of my developer friends doing this, and I was amazed. On that note, We should always learn from other domains

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/sublime_ds/now sublime-min.gif" ></center>
</div>

**So how I did this?**

Here is a step by step idea. You might want to get some data in Sublime and try it out yourself. The command that you will be using most frequently is `Cmd+Shift+L`

- Select all the text in the sublime window using `Cmd+A`
- `Cmd+Shift+L` to get the cursor on all lines
- Use `Cmd` and `Opt` with arrow keys to move these cursors to required locations. `Cmd` takes to beginning and end. `Opt` takes you token by token
- Do your Magic and write.
- Press `Delete` key to getting everything in one line
- Press `Esc` to get out from Multiple cursor mode
- Enjoy!

---

## 2. Select Selectively and Look Good while doing it:

Another functionality in Sublime that I love. We all have used Replace functionality in many text editors. This functionality is `Find and Replace` with a twist.

So, without further ado, let me demonstrate it with an example. Let's say we have a code snippet written in Python and we want to replace some word. We can very well do it with `Find and Replace` Functionality. We will find and replace each word and would end up clicking a lot of times. Sublime makes it so much easier. And it looks impressive too. *You look like you know what you are doing, which will get a few brownie points in my book.*

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/sublime_ds/sublime_mul-min.gif" ></center>
</div>

**So how I did this?**

- Select the word you want to replace
- Press `Cmd+D` multiple times to only select instances of the word you want to remove.
- When all words are selected, write the new word
- And that's all

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/sublime_ds/thor_tools_2x.png" ></center>
</div>

This concludes my post about one of the most efficient editors I have ever known. You can try to do a lot of things with Sublime but the above use cases are the ones which I find most useful. ***These simple commands will make your work much more efficient and remove the manual drudgery which is sometimes a big part of our jobs.*** Hope you end up using this in your Workflow. Trust me you will end up loving it.

---

Let me know if you liked this post. I will continue writing such Tips and Tricks in a series of posts. Also, do [follow me on Medium](https://mlwhiz.medium.com/) to get notified about my future posts.

---

*PS1: All the things above will also work with Atom text editor using the exact same commands on Mac.*

*PS2: For Window Users, Replace `Cmd` by `Ctrl` and `Opt` with `Alt` to get the same functionality.*
