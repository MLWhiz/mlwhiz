
---
title:   Five Dunder Methods in Python you should know about
date:  2021-02-03
draft: false
url : blog/2021/02/03/dunder-methods-python/
slug: dunder-methods-python

Keywords:
- magic methods in python
- dunder methods
- OOP Python

Tags:
- Python
- Programming

Categories:
- Programming
- Data Science


description: In this post, I would talk about five of the most used magic functions or "Dunder" methods.

thumbnail : /images/dunder-methods-python/main.png
image : /images/dunder-methods-python/main.png
toc : false
type : "post"
---

In my last post, I talked about Object-Oriented Programming(OOP). And I specifically talked about a single magic method `__init__` which is also called as a constructor method in the OOP terminology.

The magic part of `__init__` is that it gets called whenever an object is created automatically. But it is not the only one in any sense. Python provides us with many other magic methods that we end up using without even knowing about them. Ever used `len()`, `print()` or the `[]` operator on a list? You have been using Dunder methods.

In this post, I would talk about five of the most used magic functions or "Dunder" methods.

---

### 1. Operator Magic methods: `__add__`, `__sub__`, `__mul__`, `__truediv__`, `__lt__`,`__gt__`

In my last post, I talked about how everything ranging from int,str ,float in Python is an object. We also learned how we could call methods on an object like

```py
Fname = "Rahul"

# Now, we can use various methods defined in the string class using the below syntax

Fname.lower()
```

But, as we know, we can also use the + operator to concatenate multiple strings.

```py
Lname = "Agarwal"
print(Fname + Lname)
------------------------------------------------------------------
RahulAgarwal
```

***So, Why does the addition operator work?*** I mean how does the String object knows what to do when it encounters the plus sign? How do you write it in the str class? And the same + operation happens differently in the case of integer objects. Thus ***the operator + behaves differently in the case of String and Integer***(Fancy people call this — operator overloading).

So, can we add any two objects? Let’s try to add two objects from our elementary Account class.

![](/images/dunder-methods-python/0.png)

It fails as expected since **the operand `+` is not supported for an object of type Account**. But we can add the support of `+` to our Account class using our magic method `__add__`

```py
class Account:
    def __init__(self, account_name, balance=0):
        self.account_name = account_name
        self.balance = balance
    def __add__(self,acc):
        if isinstance(acc,Account):
            return self.balance  + acc.balance
        raise Exception(f"{acc} is not of class Account")
```

Here we add a magic method `__add__` to our class which takes two arguments — `self` and `acc`. We first check if `acc` is of class `account` and if it is, we return the sum of balances when we add these accounts. If we add anything else to an account other than an object from the `Account` class, we would be shown a descriptive error. Let us try it:

![](/images/dunder-methods-python/1.png)

So, we can add any two objects. In fact, we also have different magic methods for a variety of other operators.

* `__sub__` for subtraction(-)

* `__mul__` for multiplication(*)

* `__truediv__` for division(/)

* `__eq__` for equality (==)

* `__lt__` for less than(<)

* `__gt__` for greater than(>)

* `__le__` for less than or equal to (≤)

* `__ge__` for greater than or equal to (≥)

**As a running example,** I will try to explain all these concepts by creating a class called **Complex** to handle complex numbers. Don’t worry `Complex` would just be the class's name, and I would try to keep it as simple as possible.

So below, I create a simple method `__add__` that adds two Complex numbers, or a complex number and an `int`/`float`. It first checks if the number being added is of type `int` or `float` or `Complex`. Based on the type of number being added, we do the required addition. We also use the isinstance function to check the type of the other object. Do read the comments.

```py
import math
class Complex:
    def __init__(self, re=0, im=0):
        self.re = re
        self.im = im
    def __add__(self, other):
        # If Int or Float Added, return a Complex number where float/int is added to the real part
        if isinstance(other, int) or isinstance(other, float):
            return Complex(self.re + other,self.im)
        # If Complex Number added return a new complex number having a real and complex part
        elif  isinstance(other, Complex):
            return Complex(self.re + other.re , self.im + other.im)
        else:
            raise TypeError
```

It can be used as:

```py
a = Complex(3,4)
b = Complex(4,5)
print(a+b)
```

You would now be able to understand the below code which allows us to add, subtract, multiply and divide complex numbers with themselves as well as scalars(float, int etc.). See how these methods, in turn, return a complex number. The below code also provides the functionality to compare two complex numbers using `__eq__`,`__lt__`,`__gt__`

You really don’t need to understand all of the complex number maths, but I have tried to use most of these magic methods in this particular class. Maybe read through `__add__` and `__eq__` one.

```py
import math
class Complex:
    def __init__(self, re=0, im=0):
        self.re = re
        self.im = im

    def __add__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Complex(self.re + other,self.im)
        elif  isinstance(other, Complex):
            return Complex(self.re + other.re , self.im + other.im)
        else:
            raise TypeError

    def __sub__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Complex(self.re - other,self.im)
        elif  isinstance(other, Complex):
            return Complex(self.re - other.re, self.im - other.im)
        else:
            raise TypeError

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Complex(self.re * other, self.im * other)
        elif isinstance(other, Complex):
        #   (a+bi)*(c+di) = ac + adi +bic -bd
            return Complex(self.re * other.re - self.im * other.im,
                           self.re * other.im + self.im * other.re)
        else:
            raise TypeError

    def __truediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Complex(self.re / other, self.im / other)
        elif isinstance(other, Complex):
            x = other.re
            y = other.im
            u = self.re
            v = self.im
            repart = 1/(x**2+y**2)*(u*x + v*y)
            impart = 1/(x**2+y**2)*(v*x - u*y)
            return Complex(repart,impart)
        else:
            raise TypeError

    def value(self):
        return math.sqrt(self.re**2 + self.im**2)

    def __eq__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return  self.value() == other
        elif  isinstance(other, Complex):
            return  self.value() == other.value()
        else:
            raise TypeError

    def __lt__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return  self.value() < other
        elif  isinstance(other, Complex):
            return  self.value() < other.value()
        else:
            raise TypeError

    def __gt__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return  self.value() > other
        elif  isinstance(other, Complex):
            return  self.value() > other.value()
        else:
            raise TypeError
```

Now we can use our `Complex` class as:

![](/images/dunder-methods-python/2.png)

---

### 2. But why does the complex number print as this random string? — `__str__` and `__repr__`

Ahh! You got me. This brings us to another dunder method which lets us use the print method on our object called `__str__`. The main idea again is that when we call print(object) it calls the `__str__` method in the object. Here is how we can use that with our Complex class.

```py
class Complex:
    def __init__(self, re=0, im=0):
        self.re = re
        self.im = im

    .....
    .....

    def __str__(self):
        if self.im>=0:
            return f"{self.re}+{self.im}i"
        else:
            return f"{self.re}{self.im}i"
```

We can now recheck the output:

![](/images/dunder-methods-python/3.png)

So now, our object gets printed in a better way. But still, if we try to do the below in our notebook, the `__str__` method is not called:

![](/images/dunder-methods-python/4.png)

This is because we are not printing in the above code, and thus the `__str__` method doesn’t get called. In this case, another magic method called `__repr__` gets called. We can just add this in our class to get the same result as a print(we could also have implemented it differently). It's a dunder method inside a dunder method. Pretty nice!!!

```py
def __repr__(self):
    return self.__str__()
```

---

### 3. I am getting you. So, Is that how the len method works too? `__len__`

len() is another function that works pretty much with strings, Lists and matrices, and whatnot. To use this function with our Complex numbers class, we can use the `__len__` magic method though really not a valid use case for complex numbers in this case as the return type of `__len__` needs to be an int as per the documentation.

```py
class Complex:
    def __init__(self, re=0, im=0):
        self.re = re
        self.im = im

    ......
    ......

    def __len__(self):
        # This function return type needs to be an int
        return int(math.sqrt(self.re**2 + self.im**2))
```

Here is its usage:

![](/images/dunder-methods-python/5.png)

---

### 4. But what about the Assignment operations?

We know how the + operator works with an object. But have you thought why does the below += work? For example

```py
myStr = "This blog"
otherStr = " is awesome"

myStr+=otherStr
print(myStr)
```

This brings us to another set of dunder methods called assignment dunder methods which include `__iadd__`,`__isub__`,`__imul__`,`__itruediv__` and many others.

So If we just add the following method `__iadd__` to our class, we would be able to make assignment based additions too.

```py
class Complex:
    .....
    def __iadd__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Complex(self.re + other,self.im)
        elif  isinstance(other, Complex):
            return Complex(self.re + other.re , self.im + other.im)
        else:
            raise TypeError
```

And use it as:

![](/images/dunder-methods-python/6.png)

---

### 5. Can your class object support indexing?

Sometimes objects might contain lists, and we might need to index the object to get the value from the list. To understand this, let's take a different example. Imagine you are a company that helps users trade stock. Each user will have a Daily Transaction Book that will contain information about the user's trades/transactions over the course of the day. We can implement such a class by:

```py
class TrasnsactionBook:
    def __init__(self, user_id, shares=[]):
        self.user_id = user_id
        self.shares = shares
    def add_trade(self, name , quantity, buySell):
        self.shares.append([name,quantity,buySell])
    def __getitem__(self, i):
        return self.shares[i]
```

Do you notice the `__getitem__` here? This actually allows us to use indexing on objects of this particular class using:

![](/images/dunder-methods-python/7.png)

We can get the first trade done by the user or the second one based on the index we use. This is just a simple example, but you can set your object up to get a lot more information when you use indexing.

---

### Conclusion

Python is a magical language, and there are many constructs in Python that even advanced users may not know about. Dunder methods might be very well one of them. I hope with this post, you get a good glimpse of various dunder methods that Python offers and also understand how to implement them yourself. If you want to know about even more dunder methods, do take a look at [this blog](https://rszalski.github.io/magicmethods/#comparisons) from Rafe Kettler.

If you want to learn more about [Python](https://amzn.to/2XPSiiG), I would like to call out an excellent course on Learn [Intermediate level Python](https://bit.ly/2XshreA) from the University of Michigan. Do check it out.

I am going to be writing more of such posts in the future too. Let me know what you think about the series. Follow me up at [Medium](http://mlwhiz.medium.com/) or Subscribe to my [blog](https://mlwhiz.ck.page/a9b8bda70c) to be informed about them. As always, I welcome feedback and constructive criticism and can be reached on Twitter [@mlwhiz](https://twitter.com/MLWhiz)

Also, a small disclaimer — There might be some affiliate links in this post to relevant resources, as sharing knowledge is never a bad idea.
