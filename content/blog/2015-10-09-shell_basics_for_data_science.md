---
title: "Shell Basics every Data Scientist Should know -Part I"
date:  2015-10-09
draft: false
url : blog/2015/10/09/shell_basics_for_data_science/
slug: shell_basics_for_data_science
Category: Python, Bash
Keywords:
- Bash
 - piping
 - datascience
 - tricks
 - bash tricks
 - uniq
-  cut
 - sort
-  grep
 - wc
 - head
-  tail
- cat
- awk
- sed

Tags:
- Machine Learning
- Tools
- Awesome Guides

description: This post provides an intermediate introduction to the basics of shell commands and how they can be integrated in daily data science workflow.
toc : false

Categories:
- Data Science
type : post
thumbnail: /images/category_bgs/default_bg.jpg
image: /images/category_bgs/default_bg.jpg

---
Shell Commands are powerful. And life would be like **hell without shell** is how I like to say it(And that is probably the reason that I dislike windows).

Consider a case when you have a 6 GB pipe-delimited file sitting on your laptop and you want to find out the count of distinct values in one particular column. You can probably do this in more than one way. You could put that file in a database and run SQL Commands, or you could write a python/perl script.

Probably whatever you do it won't be simpler/less time consuming than this

```bash
cat data.txt | cut -d "|" -f 1 | sort | uniq | wc -l
```

<pre style="font-size:50%; padding:7px; margin:0em;  background-color:#FFF122">30
</pre>

And this will **run way faster** than whatever you do with perl/python script.

Now this command says



*   Use the **cat** command to print/stream the contents of the file to stdout.
*   Pipe the streaming contents from our cat command to the next command **cut**.
*   The **cut** commands specifies the delimiter by the argument **-d** and the column by the argument **-f** and streams the output to stdout.
*   Pipe the streaming content to the **sort** command which sorts the input and streams only the distinct values to the stdout. It takes the argument **-u** that specifies that we only need unique values.
*   Pipe the output to the wc -l command which counts the number of lines in the input.


There is a **lot going on here** and I will try my best to ensure that **you will be able to understand most of it by the end of this Blog post**.Although I will also try to explain more advanced concepts than the above command in this post.

Now, I use shell commands extensively at my job. I will try to explain the usage of each of the commands based on use cases that I counter nearly daily at may day job as a data scientist.

## Some Basic Commands in Shell:

There are a lot of times when you just need to know a little bit about the data. You just want to see may be a couple of lines to inspect a file. One way of doing this is opening the txt/csv file in the notepad. And that is probably the best way for small files. But you could also do it in the shell using:

### 1. cat

```bash
cat data.txt
```
<pre style="font-size:50%; padding:7px; margin:0em;  background-color:#FFF122">
yearID|teamID|lgID|playerID|salary
1985|BAL|AL|murraed02|1472819
1985|BAL|AL|lynnfr01|1090000
1985|BAL|AL|ripkeca01|800000
1985|BAL|AL|lacyle01|725000
1985|BAL|AL|flanami01|641667
1985|BAL|AL|boddimi01|625000
1985|BAL|AL|stewasa01|581250
1985|BAL|AL|martide01|560000
1985|BAL|AL|roeniga01|558333
</pre>

Now the [cat](https://en.wikipedia.org/wiki/Cat_%28Unix%29) command prints the whole file in the terminal window for you.I have not shown the whole file here.

But sometimes the files will be so big that you wont be able to open them up in notepad++ or any other software utility and there the cat command will shine.

### 2\. Head and Tail

Now you might ask me why would you print the whole file in the terminal itself? Generally I won't. But I just wanted to tell you about the cat command. For the use case when you want only the top/bottom n lines of your data you will generally use the [head](https://en.wikipedia.org/wiki/Head_%28Unix%29)/[tail](https://en.wikipedia.org/wiki/Tail_%28Unix%29) commands. You can use them as below.

```bash
head data.txt
```

<pre style="font-size:50%; padding:7px; margin:0em;  background-color:#FFF122">yearID|teamID|lgID|playerID|salary
1985|BAL|AL|murraed02|1472819
1985|BAL|AL|lynnfr01|1090000
1985|BAL|AL|ripkeca01|800000
1985|BAL|AL|lacyle01|725000
1985|BAL|AL|flanami01|641667
1985|BAL|AL|boddimi01|625000
1985|BAL|AL|stewasa01|581250
1985|BAL|AL|martide01|560000
1985|BAL|AL|roeniga01|558333
</pre>

```bash
head -n 3 data.txt
```

<pre style="font-size:50%; padding:7px; margin:0em;  background-color:#FFF122">yearID|teamID|lgID|playerID|salary
1985|BAL|AL|murraed02|1472819
1985|BAL|AL|lynnfr01|1090000
</pre>

```bash
tail data.txt
```

<pre style="font-size:50%; padding:7px; margin:0em;  background-color:#FFF122">2013|WAS|NL|bernaro01|1212500
2013|WAS|NL|tracych01|1000000
2013|WAS|NL|stammcr01|875000
2013|WAS|NL|dukeza01|700000
2013|WAS|NL|espinda01|526250
2013|WAS|NL|matthry01|504500
2013|WAS|NL|lombast02|501250
2013|WAS|NL|ramoswi01|501250
2013|WAS|NL|rodrihe03|501000
2013|WAS|NL|moorety01|493000
</pre>

```bash
tail -n 2 data.txt
```

<pre style="font-size:50%; padding:7px; margin:0em;  background-color:#FFF122">2013|WAS|NL|rodrihe03|501000
2013|WAS|NL|moorety01|493000
</pre>

Notice the structure of the shell command here.

<div class="highlight">

<pre><span class="n">CommandName</span> <span class="p">[</span><span class="o">-</span><span class="n">arg1name</span><span class="p">]</span> <span class="p">[</span><span class="n">arg1value</span><span class="p">]</span> <span class="p">[</span><span class="o">-</span><span class="n">arg2name</span><span class="p">]</span> <span class="p">[</span><span class="n">arg2value</span><span class="p">]</span> <span class="n">filename</span>
</pre>

</div>

### 3\. Piping

Now we could have also written the same command as:

```bash
cat data.txt | head
```

<pre style="font-size:50%; padding:7px; margin:0em;  background-color:#FFF122">yearID|teamID|lgID|playerID|salary
1985|BAL|AL|murraed02|1472819
1985|BAL|AL|lynnfr01|1090000
1985|BAL|AL|ripkeca01|800000
1985|BAL|AL|lacyle01|725000
1985|BAL|AL|flanami01|641667
1985|BAL|AL|boddimi01|625000
1985|BAL|AL|stewasa01|581250
1985|BAL|AL|martide01|560000
1985|BAL|AL|roeniga01|558333
</pre>

This brings me to one of the most important concepts of Shell usage - [**piping**](https://en.wikipedia.org/wiki/Pipeline_%28Unix%29). You won't be able to utilize the full power the shell provides without using this concept. And the concept is actually simple.

_Just read the "|" in the command as "pass the data on to"_

So I would read the above command as:

_cat_(print) the whole data to stream, **pass the data on to** _head_ so that it can just give me the first few lines only.

So did you understood what piping did? **It is providing us a way to use our basic commands in a consecutive manner**. There are a lot of commands that are fairly basic and it lets us use these basic commands in sequence to do some fairly non trivial things.

Now let me tell you about a couple of more commands before I show you how we can **chain** them to do fairly advanced tasks.

### 4\. wc

[wc](https://en.wikipedia.org/wiki/Wc_%28Unix%29) is a fairly useful shell utility/command that lets us **count the number of lines(-l)**, **words(-w)** or **characters(-c)** in a given file

```bash
wc -l data.txt
```

<pre style="font-size:50%; padding:7px; margin:0em;  background-color:#FFF122">23957 data.txt
</pre>

### 5\. grep

You may want to print all the lines in your file which have a particular word. Or as a Data case you might like to see the salaries for the team BAL in 2000\. In this case we have printed all the lines in the file which contain "2000|BAL". [grep](https://en.wikipedia.org/wiki/Grep) is your friend.

``` bash
grep "2000|BAL" data.txt | head
```

<pre style="font-size:50%; padding:7px; margin:0em;  background-color:#FFF122">2000|BAL|AL|belleal01|12868670
2000|BAL|AL|anderbr01|7127199
2000|BAL|AL|mussimi01|6786032
2000|BAL|AL|ericksc01|6620921
2000|BAL|AL|ripkeca01|6300000
2000|BAL|AL|clarkwi02|6000000
2000|BAL|AL|johnsch04|4600000
2000|BAL|AL|timlimi01|4250000
2000|BAL|AL|deshide01|4209324
2000|BAL|AL|surhobj01|4146789
</pre>

you could also use regular expressions with grep.

### 6\. sort

You may want to [sort](https://en.wikipedia.org/wiki/Sort_%28Unix%29) your dataset on a particular column.Sort is your friend. Say you want to find out the top 10 maximum salaries given to any player in your dataset.

```bash
sort -t "|" -k 5 -r -n data.txt | head -10
```

<pre style="font-size:50%; padding:7px; margin:0em;  background-color:#FFF122">2010|NYA|AL|rodrial01|33000000
2009|NYA|AL|rodrial01|33000000
2011|NYA|AL|rodrial01|32000000
2012|NYA|AL|rodrial01|30000000
2013|NYA|AL|rodrial01|29000000
2008|NYA|AL|rodrial01|28000000
2011|LAA|AL|wellsve01|26187500
2005|NYA|AL|rodrial01|26000000
2013|PHI|NL|leecl02|25000000
2013|NYA|AL|wellsve01|24642857
</pre>

So there are certainly a lot of options in this command. Lets go through them one by one.



*   **-t**: Which delimiter to use?
*   **-k**: Which column to sort on?
*   **-n**: If you want Numerical Sorting. Dont use this option if you want Lexographical sorting.
*   **-r**: I want to sort Descending. Sorts Ascending by Default.



### 7\. cut

This command lets you select certain columns from your data. Sometimes you may want to look at just some of the columns in your data. As in you may want to look only at the year, team and salary and not the other columns. [cut](https://en.wikipedia.org/wiki/Cut_(Unix)) is the command to use.

```bash
cut -d "|" -f 1,2,5 data.txt | head
```

<pre style="font-size:50%; padding:7px; margin:0em;  background-color:#FFF122">yearID|teamID|salary
1985|BAL|1472819
1985|BAL|1090000
1985|BAL|800000
1985|BAL|725000
1985|BAL|641667
1985|BAL|625000
1985|BAL|581250
1985|BAL|560000
1985|BAL|558333
</pre>

The options are:

*   **-d**: Which delimiter to use?
*   **-f**: Which column/columns to cut?


### 8\. uniq

[uniq](https://en.wikipedia.org/wiki/Uniq) is a little bit tricky as in you will want to use this command in sequence with sort. This command removes sequential duplicates. So in conjunction with sort it can be used to get the distinct values in the data. For example if I wanted to find out 10 distinct teamIDs in data, I would use:

```bash
cat data.txt | cut -d "|" -f 2 | sort | uniq | head
```

<pre style="font-size:50%; padding:7px; margin:0em;  background-color:#FFF122">ANA
ARI
ATL
BAL
BOS
CAL
CHA
CHN
CIN
CLE
</pre>

This command could be used with argument **-c** to count the occurrence of these distinct values. Something akin to **count distinct**.

```bash
cat data.txt | cut -d "|" -f 2 | sort | uniq -c | head
```

<pre style="font-size:50%; padding:7px; margin:0em;  background-color:#FFF122">247 ANA
458 ARI
838 ATL
855 BAL
852 BOS
368 CAL
812 CHA
821 CHN
46 CIN
867 CLE
</pre>

## Some Other Utility Commands for Other Operations

Some Other command line tools that you could use without going in the specifics as the specifics are pretty hard.

### 1\. Change delimiter in a file

**Find and Replace Magic.**: You may want to replace certain characters in file with something else using the [tr](https://en.wikipedia.org/wiki/Tr_%28Unix%29) command.

```bash
cat data.txt | tr '|' ',' |  head -4
```

<pre style="font-size:50%; padding:7px; margin:0em;  background-color:#FFF122">yearID,teamID,lgID,playerID,salary
1985,BAL,AL,murraed02,1472819
1985,BAL,AL,lynnfr01,1090000
1985,BAL,AL,ripkeca01,800000
</pre>

or the [**sed**](https://en.wikipedia.org/wiki/Sed) command

```bash
cat data.txt | sed -e 's/|/,/g' | head -4
```

<pre style="font-size:50%; padding:7px; margin:0em;  background-color:#FFF122">yearID,teamID,lgID,playerID,salary
1985,BAL,AL,murraed02,1472819
1985,BAL,AL,lynnfr01,1090000
1985,BAL,AL,ripkeca01,800000
</pre>

### 2\. Sum of a column in a file

Using the [awk](https://en.wikipedia.org/wiki/AWK) command you could find the sum of column in file. Divide it by the number of lines and you can get the mean.

```bash
cat data.txt | awk -F "|" '{ sum += $5 } END { printf sum }'
```

<pre style="font-size:50%; padding:7px; margin:0em;  background-color:#FFF122">44662539172
</pre>

awk is a powerful command which is sort of a whole language in itself. Do see the wiki page for [awk](https://en.wikipedia.org/wiki/AWK) for a lot of great usecases of awk. I also wrote a post on awk as a second part in this series. Check it [HERE](http://mlwhiz.com/blog/2015/10/11/shell_basics_for_data_science_2/)

### 3\. Find the files in a directory that satisfy a certain condition

You can do this by using the find command. Lets say you want to **find all the .txt files** in the current working dir that **start with lowercase h**.

```bash
find . -name "h*.txt"
```

<pre style="font-size:50%; padding:7px; margin:0em;  background-color:#FFF122">./hamlet.txt
</pre>

To find **all .txt files starting with h regarless of case** we could use regex.

```bash
find . -name "[Hh]*.txt"
```
<pre style="font-size:50%; padding:7px; margin:0em;  background-color:#FFF122">./hamlet.txt
./Hamlet1.txt
</pre>

### 4\. Passing file list as Argument.

[xargs](https://en.wikipedia.org/wiki/Xargs) was suggested by Gaurav in the comments, so I read about it and it is actually a very nice command which you could use in a variety of use cases.

So if you just use a pipe, any command/utility receives data on STDIN (the standard input stream) as a raw pile of data that it can sort through one line at a time. However some programs don't accept their commands on standard in. For example the rm command(which is used to remove files), touch command(used to create file with a given name) or a certain python script you wrote(which takes command line arguments). They expect it to be spelled out in the arguments to the command.  

For example: rm takes a file name as a parameter on the command line like so: rm file1.txt. If I wanted to **delete all '.txt' files starting with "h/H"** from my working directory, the below command won't work because rm expects a file as an input.

```bash
find . -name "[hH]*.txt" | rm
```

<pre style="font-size:50%; padding:7px; margin:0em;  background-color:#FFF122">usage: rm [-f | -i] [-dPRrvW] file ...
unlink file
</pre>

To get around it we can use the xargs command which reads the STDIN stream data and converts each line into space separated arguments to the command.

```bash
find . -name "[hH]*.txt" | xargs
```

<pre style="font-size:50%; padding:7px; margin:0em;  background-color:#FFF122">./hamlet.txt ./Hamlet1.txt
</pre>

Now you could use rm to remove all .txt files that start with h/H. A word of advice: Always see the output of xargs first before using rm.

```bash
find . -name "[hH]*.txt" | xargs rm
```

Another usage of xargs could be in conjunction with grep to **find all files that contain a given string**.

```bash
find . -name "*.txt" | xargs grep 'honest soldier'
```

<pre style="font-size:50%; padding:7px; margin:0em;  background-color:#FFF122">./Data1.txt:O, farewell, honest soldier;
./Data2.txt:O, farewell, honest soldier;
./Data3.txt:O, farewell, honest soldier;
</pre>

Hopefully You could come up with varied uses building up on these examples. One other use case could be to use this for **passing arguments to a python script**.  

## Other Cool Tricks

Sometimes you want your data that you got by some command line utility(Shell commands/ Python scripts) not to be shown on stdout but stored in a textfile. You can use the **">"** operator for that. For Example: You could have stored the file after replacing the delimiters in the previous example into anther file called newdata.txt as follows:

```bash
cat data.txt | tr '|' ',' > newdata.txt
```

I really got confused between **"|"** (piping) and **">"** (to_file) operations a lot in the beginning. One way to remember is that you should only use **">"** when you want to write something to a file. **"|" cannot be used to write to a file.** Another operation you should know about is the **">>"** operation. It is analogous to **">"** but it appends to an existing file rather that replacing the file and writing over.

If you would like to know more about commandline, which I guess you would, here are some books that I would recommend for a beginner:

[![](http://ws-na.amazon-adsystem.com/widgets/q?_encoding=UTF8&ASIN=1593273894&Format=_SL250_&ID=AsinImage&MarketPlace=US&ServiceVersion=20070822&WS=1&tag=mlwhizcon-20)](http://www.amazon.com/gp/product/1593273894/ref=as_li_tl?ie=UTF8&camp=1789&creative=9325&creativeASIN=1593273894&linkCode=as2&tag=mlwhizcon-20&linkId=IXZOHV6FHPTYCBCT)![](http://ir-na.amazon-adsystem.com/e/ir?t=mlwhizcon-20&l=as2&o=1&a=1593273894) [![](http://ws-na.amazon-adsystem.com/widgets/q?_encoding=UTF8&ASIN=0596009658&Format=_SL250_&ID=AsinImage&MarketPlace=US&ServiceVersion=20070822&WS=1&tag=mlwhizcon-20)](http://www.amazon.com/gp/product/0596009658/ref=as_li_tl?ie=UTF8&camp=1789&creative=9325&creativeASIN=0596009658&linkCode=as2&tag=mlwhizcon-20&linkId=2ZHHZIAJBFW3BFF7)![](http://ir-na.amazon-adsystem.com/e/ir?t=mlwhizcon-20&l=as2&o=1&a=0596009658)

The first book is more of a fun read at leisure type of book. THe second book is a little more serious. Whatever suits you.

So, this is just the tip of the iceberg. Although I am not an expert in shell usage, these commands reduced my workload to a large extent. If there are some shell commands you use on a regular basis or some shell command that are cool, do tell in the comments. I would love to include it in the blogpost.

I wrote a blogpost on awk as a second part of this post. Check it <a href="http://mlwhiz.com/blog/2015/10/11/shell_basics_for_data_science_2/" target="_blank">Here</a>
