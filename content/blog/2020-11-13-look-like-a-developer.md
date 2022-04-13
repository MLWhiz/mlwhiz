---
title:   How to Look like a 10x developer
date:  2020-11-13
draft: false
url : blog/2020/11/13/look-like-a-developer/
slug: look-like-a-developer

Keywords:
- transformers from scratch
- create own transformer
- understanding transformers
- layman transformers
- layman transformers guide

Keywords:
- data science
- productivity tools
- ohmyzsh
- powershell
- excel
- tools

Categories:
- Programming

Tags:
- Data Science
- Tools

description: I promised myself that I would have the best terminal around and I will customize the shit out of it. Then people will know how cool I am and I will take centre stage to all that is worth talking about. But, alas — one thing that doesn’t come out of the box with the terminal is the customization, and I am just lazy like every developer around. I needed something quick to move from just the sassiest looking terminal below

thumbnail : /images/look-like-a-developer/main.png
image : /images/look-like-a-developer/main.png
toc : false
type : "post"
---


I love working with shell commands. They are fast and provide a ton of flexibility to do ad-hoc things. But the thing I like most about them — Oh, they look so cool.

Or so I thought until one day everyone around me was using shell commands. I just hated that development. The coolness had gone and working with that black screen was getting a little boring day by day. I had to fulfil my innate need to stand out. I needed that source of inspiration while shipping out that next code.

It’s not a cafe I am working at? Alas, people around me were getting to know that it was all pretty generic stuff I was working on.

**So, I asked myself once again — how do I look cool while working? **The question bugging every developer, since the existence of developers. And just like the developer I am, I came up with the same old answer —** Customization.**

*I promised myself that I would have the best terminal around and I will customize the s**t out of it*. Then people will know how cool I am and I will take centre stage to all that is worth talking about. But, alas — one thing that doesn’t come out of the box with the terminal is the customization, and I am just lazy like every developer around. I needed something quick to move from just the sassiest looking terminal below.

![](/images/look-like-a-developer/0.png)

But again, as it is with everything that relates to shell —Something being hard to do *is not a bug, but a feature*. It is this feature of the shell that has let us developers look cool for decades, so I need to respect that. It makes it a whole lot easier for developers to attain the highest goal — Look cool. In the developer think, not having anything extra means that you can customize something your own way with just the things you need to have.

So, by the end of this post, which is a walkthrough on how I made my terminal look awesome, I will be having the terminal look like below which shows git statuses and has a pretty great theme.

![](/images/look-like-a-developer/1.png)

---
## How?

I don’t think this post is going to be widely circulated just like every one of my posts, so I guess I can tell you the great secret of looking cool without any harm to my own coolness*. Grins and Whispers. *It’s actually pretty simple. I just ran some commands.

But, let us take a step back before we go forward to understand what’s going on. Your Mac ships with its own version of the Terminal app. I won’t be using that. While the app is great, I have become accustomed to using iTerm2 because of all the customization options and themes it provides.

To install this theme and to have an iTerm2 terminal, you will first of all need to install iTerm2 to your MAC. You can do that [here](https://www.iterm2.com/downloads.html) on the iTerm2 downloads page. Once you have installed iTerm2, we will go through a series of steps to make terminal great again.

You need to do all these steps in our new iTerm2 terminal. I start by creating a directory to do all our work in and installing oh-my-zsh on the command line. What is oh-my-zsh? You can think of zsh shell as similar to bash shell and oh-my-zsh is a framework for maintaining your .zshrc file which is quite similar to a .bashrc file.

    mkdir iterm_theming
    cd iterm_theming
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"

This will install oh-my-zsh with its default theme for you and your terminal after a restart should look like:

![](/images/look-like-a-developer/2.png)

This seems like an improvement but not by much. Let’s change the theme now. We have various options when it comes to themes. You can get a whole list of options at this [theme page](https://github.com/ohmyzsh/ohmyzsh/wiki/Themes) for oh-my-zsh.

I will be using the [powerlevel10k](https://github.com/romkatv/powerlevel10k) theme as it itself provides a lot of configuration options, but you can essentially use any theme. First, we need to run:

    git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k

And then changeZSH_THEME=”powerlevel10k/powerlevel10k" in the ~/.zshrc file by editing it with any editor. I used nano.

Once you restart your iTerm2 session, you will be greeted with:

![Author Image: Powershell configuration Screen 1](/images/look-like-a-developer/3.png "Author Image: Powershell configuration Screen 1")

Once you push y, you will see the below window which asks you to restart iTerm2

![Author Image: Powershell configuration Screen 2](/images/look-like-a-developer/4.png "Author Image: Powershell configuration Screen 2")

Once you restart your iTerm2 again, you will be greeted with a wizard where you can choose your preferences for configuring the colour and styling of your prompt.

![](/images/look-like-a-developer/5.gif)

I went through this step multiple times to get the right style I wanted. If you don’t like a style you set you can restart the widget using the p10k configure command.

---
## The Final Look

Here is what I got finally as my styling. And here is how my workflow looks when I work with my blog.

![Author Image: Working with Terminal](/images/look-like-a-developer/6.png "Author Image: Working with Terminal")

As you can notice, it shows us a lot of information on the prompt itself by using colours and symbols in a fairly intuitive way.

*For example,* The yellow colour in the prompt shows us that the git repo is not in sync with the master. The number !5 shows the number of files changed and not yet staged for commit. The number ?3 shows the number of untracked files which we should add using the git add command. And all this without using the git status command. You can see that our branch is ahead of the master by 1 commit just by seeing ⇡1 in the green prompt. On the right-hand side, we can see the execution times of commands that took some time. You can find [descriptions](https://github.com/romkatv/powerlevel10k#what-do-different-symbols-in-git-status-mean) of all the symbols on the theme page itself.

I also keep the whole path in the prompt so that I can get it when needed. Though there is a small catch. You can see that the path gets shortened below to ~/w/mlwhiz rather than ~/web/mlwhiz. This is not again a bug but a feature to save space. The trick is to copy-paste this on prompt and press tab. You will get the whole path again.

![Author Image: Working with Terminal 2](/images/look-like-a-developer/7.png "Author Image: Working with Terminal 2")

---

## Add even more Functionality

This is not all. Apart from getting a beautiful and highly functional terminal, you also get a lot of [plugins](https://github.com/ohmyzsh/ohmyzsh/wiki/Plugins) too with oh-my-zsh. Above we have just used the git plugin. But you can choose from a long list of plugins for yourself and to try one you just need to install it by adding the plugin name to the plugins list in the .zshrc file.

For example, You can add the plugin [vscode](https://github.com/ohmyzsh/ohmyzsh/tree/master/plugins/vscode)(which provides a lot of vscode aliases) and [tmux](https://github.com/ohmyzsh/ohmyzsh/tree/master/plugins/tmux)(which provides a lot of tmux aliases) to your terminal by changing plugins=(git) to plugins=(git vscode tmux) and restarting the terminal.

So what are you waiting for? Make your terminal great again.

---

## Continue Learning

Do read some of my previous articles on working with shell commands also if you like to get better at working with the shell.
[Impress Onlookers with your newly acquired Shell Skills](https://towardsdatascience.com/impress-onlookers-with-your-newly-acquired-shell-skills-a02effb420c2)

If you would like to know more about the command line, which I guess you would, there is [The UNIX workbench](https://coursera.pxf.io/kj1N30) course on Coursera which you can try out.

Thanks for the read. I am going to be writing more beginner-friendly posts in the future too. Follow me up at [Medium](https://mlwhiz.medium.com/?source=post_page---------------------------) or Subscribe to my [blog](https://mlwhiz.ck.page/a9b8bda70c) to be informed about them.

Also, a small disclaimer — There might be some affiliate links in this post to relevant resources, as sharing knowledge is never a bad idea.
