
# Python Pro Tip: Use Iterators, Generators, and Generator Expressions

Python Pro Tip: Use Iterators, Generators, and Generator Expressions

### Understanding is vital when it comes to coding

![Photo by [Aziz Acharki](https://unsplash.com/@acharki95?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)](https://cdn-images-1.medium.com/max/10224/0*pIRWATpfX-25GJqs)*Photo by [Aziz Acharki](https://unsplash.com/@acharki95?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)*

[Python](https://amzn.to/2XPSiiG) in many ways has made our life easier when it comes to programming.

With its many libraries and functionalities, sometimes we forget to focus on some of the useful things it offers.

One of such functionalities are generators and generator expressions. I stalled learning about them for a long time but they are useful.

Have you ever encountered yield in Python code and didn’t knew what it meant? or what does an iterator or a generator means and why we use it? Or have you used ImageDataGenerator* *while working with Keras and didn’t understand what is going at the backend? Then this post is for you.

In this series of posts named **‘[Python Shorts](https://towardsdatascience.com/tagged/python-shorts),’** I will explain some simple constructs provided by Python, some essential tips and some use cases I come up with regularly in my Data Science work.

***This post is about explaining some of the difficult concepts in an easy to understand way.***

## The Problem Statement:

![](https://cdn-images-1.medium.com/max/2560/1*vb535HYa9K_XNYPvH8itLw.png)

***Let us say that we need to run a for loop over 10 Million Prime numbers.***

*I am using prime numbers in this case for understanding but it could be extended to a case where we have to process a lot of images or files in a database or big data.*

***How would you proceed with such a problem?***

Simple. We can create a list and keep all the prime numbers there.

Really? ***Think of the memory such a list would occupy.***

It would be great if we had something that could just keep the last prime number we have checked and returns just the next prime number.

That is where iterators could help us.

## The Iterator Solution

We create a class named primes and use it to generate primes.

    def check_prime(number):
        for divisor in range(2, int(number ** 0.5) + 1):
            if number % divisor == 0:
                return False
        return True

    class Primes:
        def __init__(self, max):
            # the maximum number of primes we want generated
            self.max = max
            # start with this number to check if it is a prime.
            self.number = 1
            # No of primes generated yet. We want to StopIteration when it reaches max
            self.primes_generated = 0
        def __iter__(self):
            return self
        def __next__(self):
            self.number += 1
            if self.primes_generated >= self.max:
                raise StopIteration
            elif check_prime(self.number):
                self.primes_generated+=1
                return self.number
            else:
                return self.__next__()

We can then use this as:

    prime_generator = Primes(10000000)

    for x in prime_generator:
        # Process Here

Here I have defined an iterator. This is how most of the functions like xrange or ImageGenerator work.

Every iterator needs to have:

1. an __iter__ method that returns self, and

1. an __next__ method that returns the next value.

1. a StopIteration exception that signifies the ending of the iterator.

Every iterator takes the above form and we can tweak the functions to our liking in this boilerplate code to do what we want to do.

See that we don’t keep all the prime numbers in memory just the state of the iterator like

* what max prime number we have returned and

* how many primes we have returned already.

But it seems a little too much code. Can we do better?

## The Generator Solution

![Simple yet beautiful..](https://cdn-images-1.medium.com/max/8090/0*GAYh2QEnecjFglsE)*Simple yet beautiful..*

Put simply Generators provide us ways to write iterators easily using the yield statement.

    def Primes(max):
        number = 1
        generated = 0
        while generated < max:
            number += 1
            if check_prime(number):
                generated+=1
                yield number

we can use the function as:

    prime_generator = Primes(10)
    for x in prime_generator:
        # Process Here

It is so much simpler to read. But what is yield?

We can think of yield as a return statement only as it returns the value.

But when a yield happens the state of the function is also saved in the memory. So at every iteration in for loop the function variables like number, generated and max are stored somewhere in memory.

So what is happening is that the above function is taking care of all the boilerplate code for us by using the yield statement.

Much More pythonic.

## Generator Expression Solution

![So Much Cleaner!!!](https://cdn-images-1.medium.com/max/10400/0*xgL0MvUT80_yQ6fP)*So Much Cleaner!!!*

While not explicitly better than the previous solution but we can also use Generator expression for the same task. But we might lose some functionality here. They work exactly like list comprehensions but they don’t keep the whole list in memory.

    primes = (i for i in range(1,100000000) if check_prime(i))

    for x in primes:
        # do something

Functionality loss: We can generate primes till 10M. But we can’t generate 10M primes. One can only do so much with generator expressions.

But generator expressions let us do some pretty cool things.

***Let us say we wanted to have all Pythagorean Triplets lower than 1000.***

How can we get it?

Using a generator, now we know how to use them.

    def triplet(n): # Find all the Pythagorean triplets between 1 and n
        for a in range(n):
            for b in range(a):
                for c in range(b):
                    if a*a == b*b + c*c:
                        yield(a, b, c)

We can use this as:

    triplet_generator = triplet(1000)

    for x in triplet_generator:
        print(x)

    ------------------------------------------------------------
    (5, 4, 3)
    (10, 8, 6)
    (13, 12, 5)
    (15, 12, 9)
    .....

Or, we could also have used a generator expression here:

    triplet_generator = ((a,b,c) for a in range(1000) for b in range(a) for c in range(b) if a*a == b*b + c*c)

    for x in triplet_generator:
        print(x)

    ------------------------------------------------------------
    (5, 4, 3)
    (10, 8, 6)
    (13, 12, 5)
    (15, 12, 9)
    .....
> # *Isn’t Python Beautiful?*

I hate medium code blocks sometimes due to code breaks and copy paste issues. So you can see all the code in this [kaggle kernel.](https://www.kaggle.com/mlwhiz/iterators-and-generators/)

## Conclusion

***We must always try to reduce the memory footprint in [Python](https://amzn.to/2XPSiiG). ***Iterators and generators provide us with a way to do that with Lazy evaluation.

***How do we choose which one to use?*** What we can do with generator expressions we could have done with generators or iterators too.

There is no correct answer here. Whenever I face such a dilemma, I always think in the terms of functionality vs readability. Generally,

Functionality wise: Iterators>Generators>Generator Expressions.

Readability wise: Iterators<Generators<Generator Expressions.

It is not necessary that you end up using them in your code now. But I guess understanding how these things work helps mitigate some of the confusion and panic one faces whenever these constructs come up.
> # Understanding is vital when it comes to coding

Also if you want to learn more about Python 3, I would like to call out an excellent course on Learn [Intermediate level Python](https://bit.ly/2XshreA) from the University of Michigan. Do check it out.

I am going to be writing more beginner friendly posts in the future too. Let me know what you think about the series. Follow me up at [**Medium](https://medium.com/@rahul_agarwal)** or Subscribe to my [**blog](http://eepurl.com/dbQnuX)** to be informed about them. As always, I welcome feedback and constructive criticism and can be reached on Twitter [@mlwhiz](https://twitter.com/MLWhiz).
