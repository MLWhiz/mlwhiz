---
title:  Create an Awesome Development Setup for Data Science using Atom
date:  "2020-09-02"
draft: false
url : blog/2020/09/02/atom_for_data_science/
slug: atom_for_data_science
Category: Python

Keywords:
- Pandas
- Statistics

Categories:
- Data Science
- Awesome Guides

Tags:
- Machine Learning
- Data Science
- Tools
- Productivity
- Awesome Guides

description:

thumbnail : /images/atom_for_data_science/main.gif
image : /images/atom_for_data_science/main.gif
toc : false
type: "post"
---

Before I even begin this article, let me just say that I love iPython Notebooks, and Atom is not an alternative to Jupyter in any way. Notebooks provide me an interface where I have to think of *“Coding one code block at a time,”* as I like to call it, and it helps me to think more clearly while helping me make my code more modular.

**Yet, Jupyter is not suitable for some tasks in its present form**. And the most prominent is when I have to work with .py files. And one will need to work with .py files whenever they want to push your code to production or change other people’s code. So, until now, I used sublime text to edit Python files, and I found it excellent. But recently, when I looked at the Atom editor, my loyalties seemed to shift when I saw the multiple out of the box options provided by it.

Now, the real power to Atom comes from the various packages you can install. In this post, I will talk about the packages that help make Atom just the most hackable and wholesome development environment ever.

---

## Installing Atom and Some Starting Tweaks

Before we even begin, we need to install Atom. You can do it from the main website [here](https://atom.io/.). The installation process is pretty simple, whatever your platform is. For Linux, I just downloaded the .deb file and double-clicked it. Once you have installed Atom, You can look at doing some tweaks:

* Open Core settings in Atom using `Ctrl+Shift+P` and typing settings therein. This `Ctrl+Shift+P` command is going to be one of the most important commands in Atom as it lets you navigate and run a lot of commands.

![Accessing the Settings window using `Ctrl+Shift+P`](/images/atom_for_data_science/0.png)*Accessing the Settings window using `Ctrl+Shift+P`*

* Now go to the Editor menu and Uncheck “Soft Tabs”. This is done so that TAB key registers as a TAB and not two spaces. If you want you can also activate “Soft Wrap” which wraps the text if the text exceeds the window width.

![My preferred settings for soft-wrap and soft-tabs.](/images/atom_for_data_science/1.png)*My preferred settings for soft-wrap and soft-tabs.*

Now, as we have Atom installed, we can look at some of the most awesome packages it provides. And the most important of them is GitHub.

---

### 1. Commit to Github without leaving Editor

Are you fed up with leaving your text editor to use terminal every time you push a commit to Github? If your answer is yes, Atom solves this very problem by letting you push commits without you ever leaving the text editor window.

This is one of the main features that pushed me towards Atom from Sublime Text. I like how this functionality comes preloaded with Atom and it doesn’t take much time to set it up.

To start using it, click on the GitHub link in the right bottom of the Atom screen, and the Atom screen will prompt you to log in to your Github to provide access. It is a one-time setup, and once you log in and give the token generated to Atom, you will be able to push your commits from the Atom screen itself without navigating to the terminal window.

<table><tr>
<td>
![]<img src=/images/atom_for_data_science/2.png>
</td>
<td>
![]<img src=/images/atom_for_data_science/3.png>
</td>
</tr>
</table>

The process to push a commit is:

* Change any file or multiple files.

* Click on Git on the bottom right corner.

* Stage the Changes

* Write a commit message.

* Click on Push in the bottom right corner.

* And we are done:)

Below, I am pushing a very simple commit to Github, where I add a title to my Markdown file. Its a GIF file, so it might take some time to load.

![Committing in Atom](/images/atom_for_data_science/4.gif)*Committing in Atom*


---

### 2. Write Markdown with real-time preview

I am always torn between the medium editor vs. Markdown whenever I write blog posts for my site. For one, I prefer using Markdown when I have to use Math symbols for my post or have to use custom HTML. But, I also like the Medium editor as it is WYSIWYG(What You See Is What You Get). And with Atom, I have finally found the perfect markdown editor for me, which provides me with Markdown as well as WYSIWYG. And it has now become a default option for me to create any README.md files for GitHub.

Using Markdown in Atom is again a piece of cake and is activated by default. To see a live preview with Markdown in Atom:

* Use `Ctrl+Shift+M` to open Markdown Preview Pane.

* Whatever changes you do in the document will reflect near real-time in the preview window.

![Markdown Split Screen editor](/images/atom_for_data_science/5.gif)*Markdown Split Screen editor*

### 3. Minimap — A navigation map for Large code files

Till now, we haven’t installed any new package to Atom, so let’s install an elementary package as our first package. This package is called [minimap](https://atom.io/packages/minimap), and it is something that I like to have from my Sublime Text days. It lets you have a side panel where you can click and reach any part of the code. Pretty useful for large files.

To install a package, you can go to settings and click on Install Packages.
`Ctrl+Shift+P > Settings > + Install > Minimap> Install`

![Installing Minimap or any package](/images/atom_for_data_science/6.gif)*Installing Minimap or any package*

Once you install the package, you can see the minimap on the side of your screen.

![Sidebar to navigate large files with ease](/images/atom_for_data_science/7.gif)*Sidebar to navigate large files with ease*

### 4. Python Autocomplete with function definitions in Text Editor

An editor is never really complete until it provides you with some autocomplete options for your favorite language. Atom integrates well with Kite, which tries to integrate AI and autocomplete.

So, to enable autocomplete with Kite, we can use the package named [autocomplete-python](https://atom.io/packages/autocomplete-python) in Atom. The install steps remain the same as before. i.e.

`Ctrl+Shift+P > Settings > + Install > autocomplete-python> Install`

You will also see the option of using Kite along with it. I usually end up using Kite instead of Jedi(Another autocomplete option). This is how it looks when you work on a Python document with Kite autocompletion.

![Autocomplete with Kite lets you see function definitions too.](/images/atom_for_data_science/8.gif)*Autocomplete with Kite lets you see function definitions too.*

### 5. Hydrogen — Run Python code in Jupyter environment

Want to run Python also in your Atom Editor with any Jupyter Kernel? There is a way for that too. We just need to install “[Hydrogen](https://atom.io/packages/hydrogen)” using the same method as before. Once Hydrogen is installed you can use it by:

* Run the command on which your cursor is on using Ctrl+Enter.

* Select any Kernel from the Kernel Selection Screen. I select pyt kernel from the list.

* Now I can continue working in pyt kernel.

![Runnin command using Ctrl+Enter will ask you which environment to use.](/images/atom_for_data_science/9.gif)*Runnin command using Ctrl+Enter will ask you which environment to use.*

Sometimes it might happen that you don’t see an environment/kernel in Atom. In such cases, you can install ipykernel to make that kernel visible to Jupyter as well as Atom.

Here is how to make a new kernel and make it visible in Jupyter/Atom:

    conda create -n exampleenv python=3.7
    conda activate exampleenv
    conda install -c anaconda ipykernel
    python -m ipykernel install --user --name=exampleenv

Once you run these commands, your kernel will be installed. You can now update the Atom’s kernel list by using:

`Ctrl+Shift+P` >Hydrogen: Update Kernels

![](/images/atom_for_data_science/10.gif)

And your kernel should now be available in your Atom editor.

### 6. Search Stack Overflow from your Text Editor

Stack Overflow is an integral part of any developer’s life. But you know what the hassle is? To leave the coding environment and go to Chrome to search for every simple thing you need to do. And we end up doing it back and forth throughout the day. So, what if we can access Stack Overflow from Atom? You can do precisely that through the **“[ask-stack](https://atom.io/packages/ask-stack)”** package, which lets one search for questions on SO. We can access it using `Ctrl+Alt+A`

![Access Stack Overflow in Atom using Ctrl+Alt+A.](/images/atom_for_data_science/11.gif)*Access Stack Overflow in Atom using Ctrl+Alt+A.*

Some other honorable mentions of packages you could use are:

* [Teletype](https://atom.io/packages/teletype): Do Pair Coding.

* **Linter:** Checks code for Stylistic and Programmatic errors. To enable linting in Python, You can use “[linter](https://atom.io/packages/linter)” and “[python-linters](https://atom.io/packages/python-linters)”.

* [Highlight Selected](https://atom.io/packages/highlight-selected): Highlight all occurrences of a text by double-clicking or selecting the text with a cursor.

* [Atom-File-Icons](https://atom.io/packages/atom-file-icons): Provides you with file icons in the left side tree view. Looks much better than before, right?

![Icons for files](/images/atom_for_data_science/12.png)*Icons for files*

---
## Conclusion

In this post, I talked about how I use Atom in my Python Development flow.

There are a plethora of other [packages](https://atom.io/packages) in Atom which you may like, and you can look at them to make your environment even more customizable. Or one can even write their own packages as well as Atom is called as the ***“Most Hackable Editor”***.

If you want to learn about Python and not exactly a Python editor, I would like to call out an excellent course on Learn [Intermediate level Python](https://coursera.pxf.io/0JMOOY) from the University of Michigan. Do check it out. Also, here are my [course recommendations](https://towardsdatascience.com/top-10-resources-to-become-a-data-scientist-in-2020-99a315194701?source=---------2------------------) to become a Data Scientist in 2020.

I am going to be writing more beginner-friendly posts in the future too. Follow me up at [Medium](https://mlwhiz.medium.com/) or Subscribe to my [blog](https://mlwhiz.ck.page/a9b8bda70c).

Also, a small disclaimer — There might be some affiliate links in this post to relevant resources, as sharing knowledge is never a bad idea.
