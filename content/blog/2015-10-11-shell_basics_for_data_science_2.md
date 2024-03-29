---
title: "Shell Basics every Data Scientist Should know - Part II(AWK)"
date:  2015-10-11
draft: false
url : blog/2015/10/11/shell_basics_for_data_science_2/
slug: shell_basics_for_data_science_2
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
- Bash
- awk
- sed
- groupbyawk
- filter awk

Tags:
- Machine Learning
- Tools
- Awesome Guides

description: This post provides an intermediate introduction to the basics of awk and how it can be integrated into daily data science work-flow.
toc : false
Categories:
- Data Science
type : post
thumbnail: /images/category_bgs/default_bg.jpg
image: /images/category_bgs/default_bg.jpg
---

Yesterday I got introduced to awk programming on the shell and is it cool. It lets you do stuff on the command line which you never imagined. As a matter of fact, it's a whole data analytics software in itself when you think about it. You can do selections, groupby, mean, median, sum, duplication, append. You just ask. There is no limit actually.

And it is easy to learn.

In this post, I will try to give you a brief intro about how you could add awk to your daily work-flow.

Please see my previous <a href="http://mlwhiz.com/blog/2015/10/09/shell_basics_for_data_science/" target="_blank" >post</a> if you want some background or some basic to intermediate understanding of shell commands.

## Basics/ Fundamentals

So let me start with an example first. Say you wanted to sum a column in a comma delimited file. How would you do that in shell?

Here is the command. The great thing about awk is that it took me nearly 5 sec to write this command. I did not have to open any text editor to write a python script.

It lets you do adhoc work quickly.

```bash
awk 'BEGIN{ sum=0; FS=","} { sum += $5 } END { print sum }' data.txt
```

<pre style="font-size:50%; padding:7px; margin:0em;  background-color:#FFF112">44662539172
</pre>

See the command one more time. There is a basic structure to the awk command

<div class="highlight">

<pre><span class="n">BEGIN</span> <span class="p">{</span><span class="n">action</span><span class="p">}</span>
<span class="n">pattern</span> <span class="p">{</span><span class="n">action</span><span class="p">}</span>
<span class="n">pattern</span> <span class="p">{</span><span class="n">action</span><span class="p">}</span>
<span class="p">.</span>
<span class="p">.</span>
<span class="n">pattern</span> <span class="p">{</span> <span class="n">action</span><span class="p">}</span>
<span class="n">END</span> <span class="p">{</span><span class="n">action</span><span class="p">}</span>
</pre>

</div>

An awk program consists of:

*   An optional BEGIN segment : In the begin part we initialize our variables before we even start reading from the file or the standard input.

*   pattern - action pairs: In the middle part we Process the input data. You put multiple pattern action pairs when you want to do multiple things with the same line.

*   An optional END segment: In the end part we do something we want to do when we have reached the end of file.

An awk command is called on a file using:

```bash
awk 'BEGIN{SOMETHING HERE} {SOMETHING HERE: could put Multiple Blocks Like this} END {SOMETHING HERE}' file.txt
```

You also need to know about these preinitialized variables that awk keeps track of.:

1.  FS : field separator. Default is whitespace (1 or more spaces or tabs). If you are using any other seperator in the file you should specify it in the Begin Part.
2.  RS : record separator. Default record separator is newline. Can be changed in BEGIN action.
3.  NR : NR is the variable whose value is the number of the current record. You normally use it in the action blocks in the middle.
4.  NF : The Number of Fields after the single line has been split up using FS.
5.  Dollar variables : awk splits up the line which is coming to it by using the given FS and keeps the split parts in the $ variables. For example column 1 is in `$1`, column 2 is in `$2`. `$0` is the string representation of the whole line. Note that if you want to access last column you don't have to count. You can just use `$NF`. For second last column you can use `$(NF-1)`. Pretty handy. Right.


So If you are with me till here, the hard part is done. Now the fun part starts. Lets look at the first awk command again and try to understand it.

```bash
awk 'BEGIN{ sum=0; FS=","} { sum += $5 } END { print sum }' data.txt
```

So there is a begin block. Remember before we read any line. We initialize sum to 0 and FS to ",".

Now as awk reads its input line by line it increments sum by the value in column 5(as specified by `$5`).

Note that there is no pattern specified here so awk will do the action for every line.

When awk has completed reading the file it prints out the sum.

**What if you wanted mean?**

We could create a cnt Variable:

```bash
awk 'BEGIN{ sum=0;cnt=0; FS=","} { sum += $5; cnt+=1 } END { print sum/cnt }' data.txt
```

<pre style="font-size:50%; padding:7px; margin:0em;  background-color:#FFF112">1.86436e+06
</pre>

or better yet, use our friend NR which bash is already keeping track of:

```bash
awk 'BEGIN{ sum=0; FS=","} { sum += $5 } END { print sum/NR }' data.txt
```

<pre style="font-size:50%; padding:7px; margin:0em;  background-color:#FFF112">1.86436e+06
</pre>

## Filter a file

In the mean and sum awk commands we did not put any pattern in our middle commands. Let us use a simple pattern now. Suppose we have a file Salaries.csv which contains:

```bash
head salaries.txt
```

<pre style="font-size:50%; padding:7px; margin:0em;  background-color:#FFF112">yearID,teamID,lgID,playerID,salary
1985,BAL,AL,murraed02,1472819
1985,BAL,AL,lynnfr01,1090000
1985,BAL,AL,ripkeca01,800000
1985,BAL,AL,lacyle01,725000
1985,BAL,AL,flanami01,641667
1985,BAL,AL,boddimi01,625000
1985,BAL,AL,stewasa01,581250
1985,BAL,AL,martide01,560000
1985,BAL,AL,roeniga01,558333
</pre>

I want to filter records for players who who earn more than 22 M in 2013 just because I want to. You just do:

```bash
awk 'BEGIN{FS=","} $5>=22000000 && $1==2013{print $0}' Salaries.csv
```

<pre style="font-size:50%; padding:7px; margin:0em;  background-color:#FFF112">2013,DET,AL,fieldpr01,23000000
2013,MIN,AL,mauerjo01,23000000
2013,NYA,AL,rodrial01,29000000
2013,NYA,AL,wellsve01,24642857
2013,NYA,AL,sabatcc01,24285714
2013,NYA,AL,teixema01,23125000
2013,PHI,NL,leecl02,25000000
2013,SFN,NL,linceti01,22250000
</pre>

Cool right. Now let me explain it a little bit. The part in the command "`$5`>=22000000 && `$1`==2013" is called a pattern. It says that print this line(`$0`) if and only if the Salary(`$5`) is more than 22M and(&&) year(`$1`) is equal to 2013\. If the incoming record(line) does not satisfy this pattern it never reaches the inner block.

So Now you could do basic Select SQL at the command line only if you had:

**The logic Operators:**

*   == equality operator; returns TRUE is both sides are equal

*   != inverse equality operator
*   && logical AND
*   || logical OR
*   ! logical NOT
*   <, >, <=, >= relational operators

**Normal Arithmetic Operators:** +, -, /, *, %, ^

**Some String Functions:** length, substr, split

## GroupBy

Now you will say: "Hey Dude SQL without groupby is incomplete". You are right and for that we can use the associative array. Lets just see the command first and then I will explain. So lets create another useless use case(or may be something useful to someone :)) We want to find out the number of records for each year in the file. i.e we want to find the distribution of years in the file. Here is the command:
```bash
awk 'BEGIN{FS=","}
    {my_array[$1]=my_array[$1]+1}
    END{
    for (k in my_array){if(k!="yearID")print k"|"my_array[k]};
    }' Salaries.csv
```
<pre style="font-size:50%; padding:7px; margin:0em;  background-color:#FFF112">1990|867
1991|685
1996|931
1997|925
...
</pre>

Now I would like to tell you a secret. You don't really need to declare the variables you want to use in awk. So you did not really needed to define sum, cnt variables before. I only did that because it is good practice. If you don't declare a user defined variable in awk, awk assumes it to be null or zero depending on the context. So in the command above we don't declare our myarray in the begin block and that is fine.

**Associative Array**: The variable myarray is actually an associative array. i.e. It stores data in a key value format.(Python dictionaries anyone). The same array could keep integer keys and String keys. For example, I can do this in a single code.

<div class="highlight">

<pre><span class="n">myarray</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span><span class="o">=</span><span class="s">"key"</span>
<span class="n">myarray</span><span class="p">[</span><span class="err">'</span><span class="n">mlwhiz</span><span class="err">'</span><span class="p">]</span> <span class="o">=</span> <span class="mi">1</span>
</pre>

</div>

**For Loop for associative arrays**: I could use a for loop to read associative array

<div class="highlight">

<pre><span class="k">for</span> <span class="p">(</span><span class="n">k</span> <span class="n">in</span> <span class="n">array</span><span class="p">)</span> <span class="p">{</span> <span class="n">DO</span> <span class="n">SOMETHING</span> <span class="p">}</span>
<span class="cp"># Assigns to k each Key of array (unordered)</span>
<span class="cp"># Element is array[k]</span>
</pre>

</div>

**If Statement**:Uses a syntax like C for the if statement. the else block is optional:

<div class="highlight">

<pre><span class="k">if</span> <span class="p">(</span><span class="n">n</span> <span class="o">></span> <span class="mi">0</span><span class="p">){</span>
  <span class="n">DO</span> <span class="n">SOMETHING</span>
  <span class="p">}</span>
<span class="k">else</span><span class="p">{</span>
  <span class="n">DO</span> <span class="n">SOMETHING</span>
  <span class="p">}</span>
</pre>

</div>

So lets dissect the above command now.

I set the File separator to "," in the beginning. I use the first column as the key of myarray. If the key exists I increment the value by 1.

At the end, I loop through all the keys and print out key value pairs separated by "|"

I know that the header line in my file contains "yearID" in column 1 and I don't want 'yearID|1' in the output. So I only print when Key is not equal to 'yearID'.

## GroupBy with case statement:

```bash
cat Salaries.csv | awk 'BEGIN{FS=","}
    $5<100000{array5["[0-100000)"]+=1}
    $5>=100000&&$5<250000{array5["[100000,250000)"]=array5["[100000,250000)"]+1}
    $5>=250000&&$5<500000{array5["[250000-500000)"]=array5["[250000-500000)"]+1}
    $5>=500000&&$5<1000000{array5["[500000-1000000)"]=array5["[500000-1000000)"]+1}
    $5>=1000000{array5["[1000000)"]=array5["[1000000)"]+1}
    END{
    print "VAR Distrib:";
    for (v in array5){print v"|"array5[v]}
    }'
```

<pre style="font-size:50%; padding:7px; margin:0em;  background-color:#FFF112">VAR Distrib:
[250000-500000)|8326
[0-100000)|2
[1000000)|23661
[100000,250000)|9480
</pre>

Here we used multiple pattern-action blocks to create a case statement.

## For The Brave:

This is a awk code that I wrote to calculate the Mean,Median,min,max and sum of a column simultaneously. Try to go through the code and understand it.I have added comments too. Think of this as an exercise. Try to run this code and play with it. You may learn some new tricks in the process. If you don't understand it do not worry. Just get started writing your own awk codes, you will be able to understand it in very little time.
```bash
    # Create a New file named A.txt to keep only the salary column.
    cat Salaries.csv | cut -d "," -f 5 > A.txt
    FILENAME="A.txt"

    # The first awk counts the number of lines which are numeric. We use a regex here to check if the column is numeric or not.
    # ';' stands for Synchronous execution i.e sort only runs after the awk is over.
    # The output of both commands are given to awk command which does the whole work.
    # So Now the first line going to the second awk is the number of lines in the file which are numeric.
    # and from the second to the end line the file is sorted.
    (awk 'BEGIN {c=0} $1 ~ /^[-0-9]*(\.[0-9]*)?$/ {c=c+1;} END {print c;}' "$FILENAME"; \
            sort -n "$FILENAME") | awk '
      BEGIN {
        c = 0;
        sum = 0;
        med1_loc = 0;
        med2_loc = 0;
        med1_val = 0;
        med2_val = 0;
        min = 0;
        max = 0;
      }

      NR==1 {
        LINES = $1
        # We check whether numlines is even or odd so that we keep only
        # the locations in the array where the median might be.
        if (LINES%2==0) {med1_loc = LINES/2-1; med2_loc = med1_loc+1;}
        if (LINES%2!=0) {med1_loc = med2_loc = (LINES-1)/2;}
      }

      $1 ~ /^[-0-9]*(\.[0-9]*)?$/  &&  NR!=1 {
        # setting min value
        if (c==0) {min = $1;}
        # middle two values in array
        if (c==med1_loc) {med1_val = $1;}
        if (c==med2_loc) {med2_val = $1;}
        c++
        sum += $1
        max = $1
      }
      END {
        ave = sum / c
        median = (med1_val + med2_val ) / 2
        print "sum:" sum
        print "count:" c
        print "mean:" ave
        print "median:" median
        print "min:" min
        print "max:" max
      }
    '

<pre style="font-size:50%; padding:7px; margin:0em;  background-color:#FFF112">sum:44662539172
count:23956
mean:1.86436e+06
median:507950
min:0
max:33000000
</pre>
```

## Endnote:

**awk** is an awesome tool and there are a lot of use-cases where it can make your life simple. There is a sort of a learning curve, but I think that it would be worth it in the long term. I have tried to give you a taste of awk and I have covered a lot of ground here in this post. To tell you a bit more there, awk is a full programming language. There are for loops, while loops, conditionals, booleans, functions and everything else that you would expect from a programming language. So you could look more still.

To learn more about awk you can use this [book](http://ir.nmu.org.ua/bitstream/handle/123456789/143548/ecf2f2d8a72e7c3cffca0036a73aeed4.pdf?sequence=1&). This book is a free resource and you could learn more about awk and use cases.

Or if you like to have your book binded and in paper like me you can buy this book, which is a gem:

[![](http://ws-na.amazon-adsystem.com/widgets/q?_encoding=UTF8&ASIN=1565922255&Format=_SL250_&ID=AsinImage&MarketPlace=US&ServiceVersion=20070822&WS=1&tag=mlwhizcon-20)](http://www.amazon.com/gp/product/1565922255/ref=as_li_tl?ie=UTF8&camp=1789&creative=9325&creativeASIN=1565922255&linkCode=as2&tag=mlwhizcon-20&linkId=YC37WW67AJHS3T6S)![](http://ir-na.amazon-adsystem.com/e/ir?t=mlwhizcon-20&l=as2&o=1&a=1565922255)

Do leave comments in case you find more use-cases for awk or if you want me to write on new use-cases. Or just comment weather you liked it or not and how I could improve as I am also new and trying to learn more of this.

Till then Ciao !!!
