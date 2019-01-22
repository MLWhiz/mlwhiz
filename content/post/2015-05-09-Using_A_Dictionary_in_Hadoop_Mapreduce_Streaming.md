---
title: "Hadoop Mapreduce Streaming Tricks and Techniques"
date:  2015-05-09
draft: false
url : blog/2015/05/09/Hadoop_Mapreduce_Streaming_Tricks_and_Techniques/
slug: Hadoop_Mapreduce_Streaming_Tricks_and_Techniques
Category: Python, Hadoop
Keywords: 
- Hadoop
- Python
Tags: 
- Big Data
- Hadoop
- Python
description: I have been using Hadoop a lot now a days and thought about writing some of the novel techniques that a user could use to get the most out of the Hadoop Ecosystem.
toc : false
---

Title: Hadoop Mapreduce Streaming Tricks and Techniques
date:  2015-05-09 13:43
comments: true
slug: Hadoop_Mapreduce_Streaming_Tricks_and_Techniques
Category: Hadoop
Tags: hadoop,python
description: Hadoop Description of hidden techniques that can be used to improve the way you use hadoop.
Keywords: Big Data, Hadoop

I have been using Hadoop a lot now a days and thought about writing some of the novel techniques that a user could use to get the most out of the Hadoop Ecosystem.


## Using Shell Scripts to run your Programs


<img src="/images/I-love-bash-1024x220.png" >

I am not a fan of large bash commands. The ones where you have to specify the whole path of the jar files and the such. <em>You can effectively organize your workflow by using shell scripts.</em> Now Shell scripts are not as formidable as they sound. We wont be doing programming perse using these shell scripts(Though they are pretty good at that too), we will just use them to store commands that we need to use sequentially.

Below is a sample of the shell script I use to run my Mapreduce Codes.

```bash
#!/bin/bash
#Defining program variables
IP="/data/input"
OP="/data/output"
HADOOP_JAR_PATH="/opt/cloudera/parcels/CDH/lib/hadoop-0.20-mapreduce/contrib/streaming/hadoop-streaming-2.0.0-mr1-cdh4.5.0.jar"
MAPPER="test_m.py"
REDUCER="test_r.py"

hadoop fs -rmr -skipTrash&nbsp;$OP
hadoop jar&nbsp;$HADOOP_JAR_PATH \
-file&nbsp;$MAPPER -mapper "python test_m.py" \
-file&nbsp;$REDUCER -reducer "python test_r.py" \
-input&nbsp;$IP -output&nbsp;$OP
```

I generally save them as test_s.sh and whenever i need to run them i simply type <code>sh test_s.sh</code>. This helps in three ways. 
<ul><li> It helps me to store hadoop commands in a manageable way. </li>
<li> It is easy to run the mapreduce code using the shell script. </li>
<li> <em><strong>If the code fails, I do not have to manually delete the output directory</strong></em></li>
</ul>

<blockquote>
<em>
The simplification of anything is always sensational.
<br></em>
<small>Gilbert K. Chesterton</small>
</blockquote>

## Using Distributed Cache to provide mapper with a dictionary

<img src="/images/Game-Of-Thrones-Wallpaper-House-Sigils-1.png">

Often times it happens that you want that your Hadoop Mapreduce program is able to access some static file. This static file could be a dictionary, could be parameters for the program or could be anything. What distributed cache does is that it provides this file to all the mapper nodes so that you can use that file in any way across all your mappers.
Now this concept although simple would help you to think about Mapreduce in a whole new light.
Lets start with an example. 
Supppose you have to create a sample Mapreduce program that reads a big file containing the information about all the characters in <a href="http://www.hbo.com/game-of-thrones">Game of Thrones</a> stored as <strong><code>"/data/characters/"</code></strong>:
<div style="width: 50%; margin: 0 auto;">
<table class="table">
<thead>
<tr>
<th>Cust_ID</th>
<th>User_Name</th>
<th>House</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>Daenerys Targaryen</td>
<td>Targaryen</td>
</tr>
<tr>
<td>2</td>
<td>Tyrion Lannister</td>
<td>Lannister</td>
</tr>
<tr>
<td>3</td>
<td>Cersei Lannister</td>
<td>Lannister</td>
</tr>
<tr >
<td>4</td>
<td>Robert Baratheon</td>
<td>Baratheon</td>
</tr>
<tr >
<td>5</td>
<td>Robb Stark</td>
<td>Stark</td>
</tr>
</tbody>
</table>
</div>

But you dont want to use the dead characters in the file for the analysis you want to do. <em>You want to count the number of living characters in Game of Thrones grouped by their House</em>. (I know its easy!!!!!)
One thing you could do is include an if statement in your Mapper Code which checks if the persons ID is 4 then exclude it from the mapper and such.
But the problem is that you would have to do it again and again for the same analysis as characters die like flies when it comes to George RR Martin.(Also where is the fun in that)
So you create a file which contains the Ids of all the dead characters at <strong><code>"/data/dead_characters.txt"</code></strong>:

<div style="width: 50%; margin: 0 auto;">
<table class="table">
<thead>
<tr>
<th>Died</th>
</tr>
</thead>
<tbody>
<tr>
<td>4</td>
</tr>
<tr>
<td>5</td>
</tr>
</tbody>
</table>
</div>

Whenever you have to run the analysis you can just add to this file and you wont have to change anything in the code.
Also sometimes this file would be long and you would not want to clutter your code with IDs and such.

So How Would we do it. 
Let's go in a step by step way around this.
We will create a shell script, a mapper script and a reducer script for this task.

## 1) Shell Script

```py
#!/bin/bash
#Defining program variables
DC="/data/dead_characters.txt"
IP="/data/characters"
OP="/data/output"
HADOOP_JAR_PATH="/opt/cloudera/parcels/CDH/lib/hadoop-0.20-mapreduce/contrib/streaming/hadoop-streaming-2.0.0-mr1-cdh4.5.0.jar"
MAPPER="got_living_m.py"
REDUCER="got_living_r.py"

hadoop jar&nbsp;$HADOOP_JAR_PATH \
-file&nbsp;$MAPPER -mapper "python got_living_m.py" \
-file&nbsp;$REDUCER -reducer "python got_living_r.py" \
-cacheFile&nbsp;$DC#ref \
-input&nbsp;$IP -output&nbsp;$OP
```

Note how we use the <code>"-cacheFile"</code> option here. We have specified that we will refer to the file that has been provided in the Distributed cache as <code>#ref</code>. 

Next is our Mapper Script.

## 2) Mapper Script

```py
import sys
dead_ids = set()

def read_cache():
	for line in open('ref'):
		id = line.strip()
		dead_ids.add(id)

read_cache()

for line in sys.stdin:
	rec = line.strip().split("|") # Split using Delimiter "|"
	id = rec[0]
    house = rec[2]
    if id not in dead_ids:
    	print "%s\t%s" % (house,1)
```

And our Reducer Script.

## 3) Reducer Script

```py
import sys
current_key = None
key = None
count = 0

for line in sys.stdin:
	line = line.strip()
	rec = line.split('\t')
	key = rec[0]	
	value = int(rec[1])
	
	if current_key == key:
		count += value
	else:
		if current_key:
			print "%s:%s" %(key,str(count))		
		current_key = key
		count = value

if current_key == key:
    print "%s:%s" %(key,str(count))	
```

This was a simple program and the output will be just what you expected and not very exciting. <em><strong>But the Technique itself solves a variety of common problems. You can use it to pass any big dictionary to your Mapreduce Program</strong></em>. Atleast thats what I use this feature mostly for.
Hope You liked it. Will try to expand this post with more tricks.

The codes for this post are posted at github <a href="https://github.com/MLWhiz/Hadoop-Mapreduce-Tricks">here</a>.

Other Great Learning Resources For Hadoop:
<ul>
<li>
<a href="http://www.google.co.in/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0CB0QFjAA&url=http%3A%2F%2Fwww.michael-noll.com%2Ftutorials%2Fwriting-an-hadoop-mapreduce-program-in-python%2F&ei=8RRVVdP2IMe0uQShsYDYBg&usg=AFQjCNH3DqrlSIG8D-K8jgQWTALic1no5A&sig2=BivwTW6mdJs5c9w9VaSK2Q&bvm=bv.93112503,d.c2E">Michael Noll's Hadoop Mapreduce Tutorial</a>
</li>
<li>
<a href="http://www.google.co.in/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=0CCMQFjAB&url=http%3A%2F%2Fhadoop.apache.org%2Fdocs%2Fr1.2.1%2Fstreaming.html&ei=8RRVVdP2IMe0uQShsYDYBg&usg=AFQjCNEIB4jmqcBs-GepHdn7DRxqTI9zXA&sig2=nYkAnDjjjaum5YVlYuMUJQ&bvm=bv.93112503,d.c2E">Apache's Hadoop Streaming Documentation</a>
</li>
</ul>

Also I like these books a lot. Must have for a Hadooper....

<div style="margin-left:1em ; text-align: center;">
<a target="_blank"  href="https://www.amazon.com/gp/product/1785887211/ref=as_li_tl?ie=UTF8&camp=1789&creative=9325&creativeASIN=1785887211&linkCode=as2&tag=mlwhizcon-20&linkId=a0e7b4f0b2ea4a5146042890e1c04f7e"><img border="0" src="//ws-na.amazon-adsystem.com/widgets/q?_encoding=UTF8&MarketPlace=US&ASIN=1785887211&ServiceVersion=20070822&ID=AsinImage&WS=1&Format=_SL250_&tag=mlwhizcon-20" ></a><img src="//ir-na.amazon-adsystem.com/e/ir?t=mlwhizcon-20&l=am2&o=1&a=1785887211" width="1" height="1" border="0" alt="" style="border:none !important; margin:0px !important;" />

</t></t>

<a target="_blank"  href="https://www.amazon.com/gp/product/1491901632/ref=as_li_tl?ie=UTF8&camp=1789&creative=9325&creativeASIN=1491901632&linkCode=as2&tag=mlwhizcon-20&linkId=4122280e94f7bbd0ceebc9d13e60d103"><img border="0" src="//ws-na.amazon-adsystem.com/widgets/q?_encoding=UTF8&MarketPlace=US&ASIN=1491901632&ServiceVersion=20070822&ID=AsinImage&WS=1&Format=_SL250_&tag=mlwhizcon-20" ></a><img src="//ir-na.amazon-adsystem.com/e/ir?t=mlwhizcon-20&l=am2&o=1&a=1491901632" width="1" height="1" border="0" alt="" style="border:none !important; margin:0px !important;" />
</div>

<p>The first book is a guide for using Hadoop as well as spark with Python. While the second one contains a detailed overview of all the things in Hadoop. Its the definitive guide.</p>

<script src="//z-na.amazon-adsystem.com/widgets/onejs?MarketPlace=US&adInstanceId=c4ca54df-6d53-4362-92c0-13cb9977639e"></script>
