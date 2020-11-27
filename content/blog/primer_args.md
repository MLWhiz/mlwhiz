---
title: "A primer on *args, **kwargs, decorators for Data Scientists"
date:  2019-05-14
draft: false
url : blog/2019/05/14/python_args_kwargs/
slug: python_args_kwargs

Categories:
- Programming
- Data Science
Keywords:
- Python
Tags:
- Python

description: This post is a part of my series on Python Shorts. Some tips on how to use python. This post is about explaining args, kwargs and decorators in an easy to understand way.
thumbnail : /images/args/magic.jpeg
image : /images/args/magic.jpeg
toc : false
type: "post"
---

Python has a lot of constructs that are reasonably easy to learn and use in our code. Then there are some constructs which always confuse us when we encounter them in our code.

Then are some that even seasoned programmers are not able to understand. `*args`, `**kwargs` and decorators are some constructs that fall into this category.

I guess a lot of my data science friends have faced them too.

Most of the seaborn functions use `*args` and `**kwargs` in some way or other.

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/args/scatterplot.png"></center>
</div>

Or what about decorators?

Every time you see a warning like some function will be deprecated in the next version. The sklearn package uses decorators for that. You can see the @deprecated in the source code. That is a decorator function.

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/args/deprecated.png"></center>
</div>

In this series of posts named Python Shorts I will explain some simple constructs provided by Python, some essential tips and some use cases I come up with regularly in my Data Science work.

***This post is about explaining some of the difficult concepts in an easy to understand way.***

## What are `*args`?

In simple terms,***you can use *args to give an arbitrary number of inputs to your function.***

### A simple example:

Let us say we have to create a function that adds two numbers. We can do this easily in python.
```py
def adder(x,y):
    return x+y
```
What if we want to create a function to add three variables?

```py
def adder(x,y,z):
    return x+y+z
```
What if we want the same function to add an unknown number of variables? Please note that we can use `*args` or `*argv` or `*anyOtherName` to do this. It is the `*` that matters.

```py

def adder(*args):
    result = 0
    for arg in args:
        result+=arg
    return result
```

What `*args` does is that it takes all your passed arguments and provides a variable length argument list to the function which you can use as you want.

Now you can use the same function as follows:

```py
adder(1,2)
adder(1,2,3)
adder(1,2,5,7,8,9,100)
```
and so on.

**Now, have you ever thought how the print function in python could take so many arguments? `*args`**

---

## What are `**kwargs`?

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/args/scatterplot.png"></center>
</div>

In simple terms,**you can use `**kwargs` to give an arbitrary number of *Keyworded inputs* to your function and access them using a dictionary.**

### A simple example:

Let’s say you want to create a print function that can take a name and age as input and print that.

```py
def myprint(name,age):
    print(f'{name} is {age} years old')
```

Simple. Let us now say you want the same function to take two names and two ages.
```py
def myprint(name1,age1,name2,age2):
    print(f'{name1} is {age1} years old')
    print(f'{name2} is {age2} years old')
```
You guessed right my next question is: ***What if I don’t know how many arguments I am going to need?***

Can I use `*args?` Guess not since name and age order is essential. We don’t want to write “28 is Michael years old”.

Come `**kwargs` in the picture.
```py
def myprint(**kwargs):
    for k,v in kwargs.items():
        print(f'{k} is {v} years old')
```

You can call this function using:
```py
myprint(Sansa=20,Tyrion=40,Arya=17)
```
```
Output:
-----------------------------------
Sansa is 20 years old
Tyrion is 40 years old
Arya is 17 years old
```
***Remember we never defined Sansa or Arya or Tyrion as our methods arguments.***

***That is a pretty powerful concept.*** And many programmers utilize this pretty cleverly when they write wrapper libraries.

For example, `seaborn.scatterplot` function wraps the `plt.scatter` function from Matplotlib. Essentially, using `*args` and `**kwargs` we can provide all the arguments that plt.scatter can take to seaborn.Scatterplot as well.

This can save a lot of coding effort and also makes the code future proof. If at any time in the future `plt.scatter` starts accepting any new arguments the seaborn.Scatterplot function will still work.

---

## What are Decorators?

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/args/decorator.jpeg"></center>
</div>

In simple terms: ***Decorators are functions that wrap another function thus modifying its behavior.***

### A simple example:

Let us say we want to add custom functionality to some of our functions. The functionality is that whenever the function gets called the “function name begins” is printed and whenever the function ends the "function name ends” and time taken by the function is printed.

Let us assume our function is:
```py
def somefunc(a,b):
    output = a+b
    return output
```

We can add some print lines to all our functions to achieve this.
```py
import time
def somefunc(a,b):
    print("somefunc begins")
    start_time = time.time()
    output = a+b
    print("somefunc ends in ",time.time()-start_time, "secs")
    return output

out = somefunc(4,5)
```

```
OUTPUT:
-------------------------------------------
somefunc begins
somefunc ends in  9.5367431640625e-07 secs
```

***But, Can we do better?***

This is where decorators excel. We can use decorators to wrap any function.
```py
from functools import wraps

def timer(func):
    [@wraps](http://twitter.com/wraps)(func)
    def wrapper(a,b):
        print(f"{func.__name__!r} begins")
        start_time = time.time()
        func(a,b)
        print(f"{func.__name__!r} ends in {time.time()-start_time}  secs")
    return wrapper
```

This is how we can define any decorator. functools helps us create decorators using wraps. In essence, we do something before any function is called and do something after a function is called in the above decorator.

We can now use this timer decorator to decorate our function somefunc

```py
@timer
def somefunc(a,b):
    output = a+b
    return output
```
Now calling this function, we get:
```py
a = somefunc(4,5)
```
```
Output
---------------------------------------------
'somefunc' begins
'somefunc' ends in 2.86102294921875e-06  secs
```

Now we can append `@timer` to each of our function for which we want to have the time printed. And we are done.

Really?

## Connecting all the pieces

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/args/lego.jpeg"></center>
</div>

***What if our function takes three arguments? Or many arguments?***

This is where whatever we have learned till now connects. We use `*args` and `**kwargs`

We change our decorator function as:

```py
from functools import wraps

def timer(func):
    [@wraps](http://twitter.com/wraps)(func)
    def wrapper(*args,**kwargs):
        print(f"{func.__name__!r} begins")
        start_time = time.time()
        func(*args,**kwargs)
        print(f"{func.__name__!r} ends in {time.time()-start_time}  secs")
    return wrapper
```
Now our function can take any number of arguments, and our decorator will still work.

> # Isn’t Python Beautiful?

In my view, decorators could be pretty helpful. I provided only one use case of decorators, but there are several ways one can use them.

You can use a decorator to debug code by checking which arguments go in a function. Or a decorator could be used to count the number of times a particular function has been called. This could help with counting recursive calls.

---

## Conclusion

In this post, I talked about some of the constructs you can find in python source code and how you can understand them.

It is not necessary that you end up using them in your code now. But I guess understanding how these things work helps mitigate some of the confusion and panic one faces whenever these constructs come up.

> # Understanding is vital when it comes to coding

Also if you want to learn more about Python 3, I would like to call out an excellent course on Learn [Intermediate level Python](https://bit.ly/2XshreA) from the University of Michigan. Do check it out.

I am going to be writing more beginner friendly posts in the future too. Let me know what you think about the series. Follow me up at [**Medium**](https://mlwhiz.medium.com/) or Subscribe to my [**blog**](https://mlwhiz.ck.page/a9b8bda70c).
