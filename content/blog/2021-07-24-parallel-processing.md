
---
title:   How Can Data Scientists Use Parallel Processing?
date:  2021-07-24
draft: false
url : blog/2021/07/24/parallel-processing/
slug: parallel-processing

Keywords:
- parallel

Tags:
- Python
- Transformers
- Programming

Categories:
- Deep Learning
- Natural Language Processing
- Awesome Guides

description: How can data scientists use parallel processing?

thumbnail : /images/parallel-processing/main.png
image : /images/parallel-processing/main.png
toc : false
type : "post"
---


Finally, my program is running! Should I go and get a coffee?

We data scientists have got powerful laptops. Laptops which have quad-core or octa-core processors and Turbo Boost technology. We routinely work with servers with even more cores and computing power. But do we really use the raw power we have at hand?

Instead of taking advantage of our resources, too often we sit around and wait for time-consuming processes to finish. Sometimes we wait for hours, even when urgent deliverables are approaching the deadline. Can we somehow do better?

***In this post, I will explain how to use multiprocessing and Joblib to make your code parallel and get out some extra work out of that big machine of yours.***

---
## 1. Using Mutiprocessing with Single parameter function:

To motivate multiprocessing, I will start with a problem where we have a big list and we want to apply a function to every element in the list.

Why do we want to do this? This might feel like a trivial problem but this is particularly what we do on a daily basis in Data Science. For Example: We have a model and we run multiple iterations of the model with different hyperparameters. Or, we are creating a new feature in a big dataframe and we apply a function row by row to a dataframe using the apply keyword. By the end of this post, you would be able to parallelize most of the use cases you face in data science with this simple construct.

So, coming back to our toy problem, let’s say we want to apply the square function to all our elements in the list.

    def square(num):
        return x**2

Of course we can use simple python to run the above function on all elements of the list.

    result = [f(x) for x in list(range(100000))]

But, the above code is running sequentially. Here is how we can use multiprocessing to apply this function to all the elements of a given list list(range(100000)) in parallel using the 8 cores in our powerful computer.

    from multiprocessing import Pool
    pool = Pool(8)
    result = pool.map(f,list(range(100000)))
    pool.close()

The lines above create a multiprocessing pool of 8 workers and we can use this pool of 8 workers to map our required function to this list.

Lets check how this code performs:

```py
from multiprocessing import Pool
import time
import plotly.express as px
import plotly
import pandas as pd

def f(x):
    return x**2

def runner(list_length):
    print(f"Size of List:{list_length}")
    t0 = time.time()
    result1 = [f(x) for x in list(range(list_length))]
    t1 = time.time()
    print(f"Without multiprocessing we ran the function in {t1 - t0:0.4f} seconds")
    time_without_multiprocessing = t1-t0
    t0 = time.time()
    pool = Pool(8)
    result2 = pool.map(f,list(range(list_length)))
    pool.close()
    t1 = time.time()
    print(f"With multiprocessing we ran the function in {t1 - t0:0.4f} seconds")
    time_with_multiprocessing = t1-t0
    return time_without_multiprocessing, time_with_multiprocessing

if __name__ ==  '__main__':
    times_taken = []
    for i in range(1,9):
        list_length = 10**i
        time_without_multiprocessing, time_with_multiprocessing = runner(list_length)
        times_taken.append([list_length, 'No Mutiproc', time_without_multiprocessing])
        times_taken.append([list_length, 'Multiproc', time_with_multiprocessing])

    timedf = pd.DataFrame(times_taken,columns = ['list_length', 'type','time_taken'])
    fig =  px.line(timedf,x = 'list_length',y='time_taken',color='type',log_x=True)
    plotly.offline.plot(fig, filename='comparison_bw_multiproc.html')

```

    Size of List:10
    Without multiprocessing we ran the function in 0.0000 seconds
    With multiprocessing we ran the function in 0.5960 seconds
    Size of List:100
    Without multiprocessing we ran the function in 0.0001 seconds
    With multiprocessing we ran the function in 0.6028 seconds
    Size of List:1000
    Without multiprocessing we ran the function in 0.0006 seconds
    With multiprocessing we ran the function in 0.6052 seconds
    Size of List:10000
    Without multiprocessing we ran the function in 0.0046 seconds
    With multiprocessing we ran the function in 0.5956 seconds
    Size of List:100000
    Without multiprocessing we ran the function in 0.0389 seconds
    With multiprocessing we ran the function in 0.6486 seconds
    Size of List:1000000
    Without multiprocessing we ran the function in 0.3654 seconds
    With multiprocessing we ran the function in 0.7684 seconds
    Size of List:10000000
    Without multiprocessing we ran the function in 3.6297 seconds
    With multiprocessing we ran the function in 1.8084 seconds
    Size of List:100000000
    Without multiprocessing we ran the function in 36.0620 seconds
    With multiprocessing we ran the function in 16.9765 seconds

![Image by Author](/images/parallel-processing/0.png)

As we can see the runtime of multiprocess was somewhat more till some list length but doesn’t increase as fast as the non-multiprocessing function runtime increases for larger list lengths. This tells us that there is a certain overhead of using multiprocessing and it doesn’t make too much sense for computations that take a small time.

In practice, we won’t be using multiprocessing for functions that get over in milliseconds but for much larger computations that could take more than a few seconds and sometimes hours. So lets try a more involved computation which would take more than 2 seconds. I am using time.sleep as a proxy for computation here.

```py

from multiprocessing import Pool
import time
import plotly.express as px
import plotly
import pandas as pd

def f(x):
    time.sleep(2)
    return x**2


def runner(list_length):
    print(f"Size of List:{list_length}")
    t0 = time.time()
    result1 = [f(x) for x in list(range(list_length))]
    t1 = time.time()
    print(f"Without multiprocessing we ran the function in {t1 - t0:0.4f} seconds")
    time_without_multiprocessing = t1-t0
    t0 = time.time()
    pool = Pool(8)
    result2 = pool.map(f,list(range(list_length)))
    pool.close()
    t1 = time.time()
    print(f"With multiprocessing we ran the function in {t1 - t0:0.4f} seconds")
    time_with_multiprocessing = t1-t0
    return time_without_multiprocessing, time_with_multiprocessing

if __name__ ==  '__main__':
    times_taken = []
    for i in range(1,10):
        list_length = i
        time_without_multiprocessing, time_with_multiprocessing = runner(list_length)
        times_taken.append([list_length, 'No Mutiproc', time_without_multiprocessing])
        times_taken.append([list_length, 'Multiproc', time_with_multiprocessing])

    timedf = pd.DataFrame(times_taken,columns = ['list_length', 'type','time_taken'])
    fig =  px.line(timedf,x = 'list_length',y='time_taken',color='type')
    plotly.offline.plot(fig, filename='comparison_bw_multiproc.html')

```

    Size of List:1
    Without multiprocessing we ran the function in 2.0012 seconds
    With multiprocessing we ran the function in 2.7370 seconds
    Size of List:2
    Without multiprocessing we ran the function in 4.0039 seconds
    With multiprocessing we ran the function in 2.6518 seconds
    Size of List:3
    Without multiprocessing we ran the function in 6.0074 seconds
    With multiprocessing we ran the function in 2.6580 seconds
    Size of List:4
    Without multiprocessing we ran the function in 8.0127 seconds
    With multiprocessing we ran the function in 2.6421 seconds
    Size of List:5
    Without multiprocessing we ran the function in 10.0173 seconds
    With multiprocessing we ran the function in 2.7109 seconds
    Size of List:6
    Without multiprocessing we ran the function in 12.0039 seconds
    With multiprocessing we ran the function in 2.6438 seconds
    Size of List:7
    Without multiprocessing we ran the function in 14.0240 seconds
    With multiprocessing we ran the function in 2.6375 seconds
    Size of List:8
    Without multiprocessing we ran the function in 16.0216 seconds
    With multiprocessing we ran the function in 2.6376 seconds
    Size of List:9
    Without multiprocessing we ran the function in 18.0183 seconds
    With multiprocessing we ran the function in 4.6141 seconds

![Image by Author](/images/parallel-processing/1.png)

As you can see, the difference is much more stark in this case and the function without multiprocess takes much more time in this case compared to when we use multiprocess. Again this makes perfect sense as when we start multiprocess 8 workers start working in parallel on the tasks while when we don’t use multiprocessing the tasks happen in a sequential manner with each task taking 2 seconds.

---
## 2. Multiprocessing with function with Multiple Params function:

An extension to the above code is the case when we have to run a function that could take multiple parameters. For a use case, let’s say you have to tune a particular model using multiple hyperparameters. You can do something like:

    import random
    def model_runner(n_estimators, max_depth):
        # Some code that runs and fits our model here using the   
        # hyperparams in the argument.
        # Proxy for this code with sleep.
        time.sleep(random.choice([1,2,3])
        # Return some model evaluation score
        return random.choice([1,2,3])

How would you run such a function. You can do this in two ways.

**a) Using Pool.map and * magic**

    def multi_run_wrapper(args):
       return model_runner(*args)

    pool = Pool(4)
    hyperparams = [[100,4],[150,5],[200,6],[300,4]]

    results = pool.map(multi_run_wrapper,hyperparams)
    pool.close()

In the above code, we provide args to the model_runner using

**b) Using pool.starmap**

From Python3.3 onwards we can use starmap method to achieve what we have done above even more easily.

    pool = Pool(4)
    hyperparams = [[100,4],[150,5],[200,6],[300,4]]

    results = pool.starmap(model_runner,hyperparams)
    pool.close()

---
## 3. Using Joblib with Single parameter function:

Joblib is another library that provides a simple helper class to write embarassingly parallel for loops using multiprocessing and I find it pretty much easier to use than the multiprocessing module. Running a parallel process is as simple as writing a single line with the Parallel and delayed keywords:

    from joblib import Parallel, delayed
    import time

    def f(x):
        time.sleep(2)
        return x**2

    results = Parallel(n_jobs=8)(delayed(f)(i) for i in range(10))

Let’s try to compare Joblib parallel to multiprocessing module using the same function we used before.

```py
from multiprocessing import Pool
import time
import plotly.express as px
import plotly
import pandas as pd
from joblib import Parallel, delayed

def f(x):
    time.sleep(2)
    return x**2


def runner(list_length):
    print(f"Size of List:{list_length}")
    t0 = time.time()
    result1 = Parallel(n_jobs=8)(delayed(f)(i) for i in range(list_length))
    t1 = time.time()
    print(f"With joblib we ran the function in {t1 - t0:0.4f} seconds")
    time_without_multiprocessing = t1-t0
    t0 = time.time()
    pool = Pool(8)
    result2 = pool.map(f,list(range(list_length)))
    pool.close()
    t1 = time.time()
    print(f"With multiprocessing we ran the function in {t1 - t0:0.4f} seconds")
    time_with_multiprocessing = t1-t0
    return time_without_multiprocessing, time_with_multiprocessing

if __name__ ==  '__main__':
    times_taken = []
    for i in range(1,16):
        list_length = i
        time_without_multiprocessing, time_with_multiprocessing = runner(list_length)
        times_taken.append([list_length, 'No Mutiproc', time_without_multiprocessing])
        times_taken.append([list_length, 'Multiproc', time_with_multiprocessing])

    timedf = pd.DataFrame(times_taken,columns = ['list_length', 'type','time_taken'])
    fig =  px.line(timedf,x = 'list_length',y='time_taken',color='type')
    plotly.offline.plot(fig, filename='comparison_bw_multiproc.html')

```

![Image by Author](/images/parallel-processing/2.png)*Image by Author*

We can see that the runtimes are pretty much comparable and the joblib code looks much more succint than that of multiprocessing.

---
## 4. Using Joblib with Multiple Params function:

Using multiple arguments for a function is as simple as just passing the arguments using Joblib. Here is a minimal example you can use.

    from joblib import Parallel, delayed
    import time

    def f(x,y):
        time.sleep(2)
        return x**2 + y**2

    params = [[x,x] for x in range(10)]
    results = Parallel(n_jobs=8)(delayed(f)(x,y) for x,y in params)

---
## Conclusion

Multiprocessing is a nice concept and something every data scientist should at least know about it. It won’t solve all your problems, and you should still work on optimizing your functions. But having it would save a lot of time you would spend just waiting for your code to finish.

### Continue Learning

If you want to learn more about [Python](https://amzn.to/2XPSiiG) 3, I would like to call out an excellent course on Learn [Intermediate level Python](https://coursera.pxf.io/0JMOOY) from the University of Michigan. Do check it out.

I am going to be writing more beginner-friendly posts in the future too. Follow me up at [Medium](http://mlwhiz.medium.com) or Subscribe to my [blog](mlwhiz.com) to be informed about them. As always, I welcome feedback and constructive criticism and can be reached on Twitter [@mlwhiz](https://twitter.com/MLWhiz).

Also, a small disclaimer — There might be some affiliate links in this post to relevant resources, as sharing knowledge is never a bad idea.
