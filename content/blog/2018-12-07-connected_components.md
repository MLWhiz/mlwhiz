---
title: "To all Data Scientists - The one Graph Algorithm you need to know"
date:   2018-12-07
draft: false
url : blog/2018/12/07/connected_components/
slug: connected_components
Category: pyspark, python, graphs
Keywords:
- connected components pyspark
-  connected components pyspark paper
-  connected components implementation
- connected components algorithm
-  connected components using bfs
-  connected components workbench
-  graph algorithms and application
-  graph algorithms data science
-  graph algorithms data structure
Tags:
- Data Science
- Graphs
- Machine Learning
- Big Data
Categories:
- Data Science
- Big Data

description: Connected components implementation in Python and pyspark
toc : false
thumbnail: https://upload.wikimedia.org/wikipedia/commons/8/85/Pseudoforest.svg
image :  https://upload.wikimedia.org/wikipedia/commons/8/85/Pseudoforest.svg
type : post
---

Graphs provide us with a very useful data structure. They can help us to find structure within our data. With the advent of Machine learning and big data we need to get as much information as possible about our data. Learning a little bit of graph theory can certainly help us with that.

Here is a [Graph Analytics for Big Data course on Coursera by UCSanDiego](https://imp.i384100.net/9LLABy) which I highly recommend to learn the basics of graph theory. You can start for free with the 7-day Free Trial.

One of the algorithms I am going to focus in the current post is called **Connected Components**. Why it is important. We all know clustering.

*You can think of Connected Components in very layman's terms as sort of a hard clustering algorithm which finds clusters/islands in related/connected data. As a concrete example: Say you have data about roads joining any two cities in the world. And you need to find out all the continents in the world and which city they contain.*

How will you achieve that? Come on give some thought.

To put a **Retail Perspective**: Lets say, we have a lot of customers using a lot of accounts. One way in which we can use the Connected components algorithm is to find out distinct families in our dataset. We can assume edges(roads) between CustomerIDs based on same credit card usage, or same address or same mobile number etc. Once we have those connections, we can then run the connected component algorithm on the same to create individual clusters to which we can then assign a family ID. We can use these family IDs to provide personalized recommendations based on a family needs. We can also use this family ID to fuel our classification algorithms by creating grouped features based on family.

In **Finance Perspective**: Another use case would be to capture fraud using these family IDs. If an account has done fraud in past, it is highly probable that the connected accounts are also susceptible to fraud.

So enough of use cases. Lets start with a simple graph class written in Python to start up our exploits with code.

This post will revolve more around code from here onwards.

```py
""" A Python Class
A simple Python graph class, demonstrating the essential
facts and functionalities of graphs.
Taken from https://www.python-course.eu/graphs_python.php
Changed the implementation a little bit to include weighted edges
"""

class Graph(object):
    def __init__(self, graph_dict=None):
        """ initializes a graph object
            If no dictionary or None is given,
            an empty dictionary will be used
        """
        if graph_dict == None:
            graph_dict = {}
        self.__graph_dict = graph_dict

    def vertices(self):
        """ returns the vertices of a graph """
        return list(self.__graph_dict.keys())

    def edges(self):
        """ returns the edges of a graph """
        return self.__generate_edges()

    def add_vertex(self, vertex):
        """ If the vertex "vertex" is not in
            self.__graph_dict, a key "vertex" with an empty
            dict as a value is added to the dictionary.
            Otherwise nothing has to be done.
        """
        if vertex not in self.__graph_dict:
            self.__graph_dict[vertex] = {}

    def add_edge(self, edge,weight=1):
        """ assumes that edge is of type set, tuple or list
        """
        edge = set(edge)
        (vertex1, vertex2) = tuple(edge)
        if vertex1 in self.__graph_dict:
            self.__graph_dict[vertex1][vertex2] = weight
        else:
            self.__graph_dict[vertex1] = {vertex2:weight}

        if vertex2 in self.__graph_dict:
            self.__graph_dict[vertex2][vertex1] = weight
        else:
            self.__graph_dict[vertex2] = {vertex1:weight}


    def __generate_edges(self):
        """ A static method generating the edges of the
            graph "graph". Edges are represented as sets
            with one (a loop back to the vertex) or two
            vertices
        """
        edges = []
        for vertex in self.__graph_dict:
            for neighbour,weight in self.__graph_dict[vertex].iteritems():
                if (neighbour, vertex, weight) not in edges:
                    edges.append([vertex, neighbour, weight])
        return edges

    def __str__(self):
        res = "vertices: "
        for k in self.__graph_dict:
            res += str(k) + " "
        res += "\nedges: "
        for edge in self.__generate_edges():
            res += str(edge) + " "
        return res

    def adj_mat(self):
        return self.__graph_dict
```

You can certainly play with our new graph class.Here we try to build some graphs.

```py
g = { "a" : {"d":2},
      "b" : {"c":2},
      "c" : {"b":5,  "d":3, "e":5}
    }
graph = Graph(g)
print("Vertices of graph:")
print(graph.vertices())
print("Edges of graph:")
print(graph.edges())
print("Add vertex:")
graph.add_vertex("z")
print("Vertices of graph:")
print(graph.vertices())
print("Add an edge:")
graph.add_edge({"a","z"})    
print("Vertices of graph:")
print(graph.vertices())
print("Edges of graph:")
print(graph.edges())
print('Adding an edge {"x","y"} with new vertices:')
graph.add_edge({"x","y"})
print("Vertices of graph:")
print(graph.vertices())
print("Edges of graph:")
print(graph.edges())
```


<pre style="font-family:courier new,monospace; background-color:#f6c6529c; color:#000000">Vertices of graph:
['a', 'c', 'b']
Edges of graph:
[['a', 'd', 2], ['c', 'b', 5], ['c', 'e', 5], ['c', 'd', 3], ['b', 'c', 2]]
Add vertex:
Vertices of graph:
['a', 'c', 'b', 'z']
Add an edge:
Vertices of graph:
['a', 'c', 'b', 'z']
Edges of graph:
[['a', 'z', 1], ['a', 'd', 2], ['c', 'b', 5], ['c', 'e', 5], ['c', 'd', 3], ['b', 'c', 2], ['z', 'a', 1]]
Adding an edge {"x","y"} with new vertices:
Vertices of graph:
['a', 'c', 'b', 'y', 'x', 'z']
Edges of graph:
[['a', 'z', 1], ['a', 'd', 2], ['c', 'b', 5], ['c', 'e', 5], ['c', 'd', 3], ['b', 'c', 2], ['y', 'x', 1], ['x', 'y', 1], ['z', 'a', 1]]
</pre>

Lets do something interesting now.

We will use the above graph class for our understanding purpose. There are many Modules in python which we can use to do whatever I am going to do next,but to understand the methods we will write everything from scratch.
Lets start with an example graph which we can use for our purpose.

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/ad/MapGermanyGraph.svg/1200px-MapGermanyGraph.svg.png"  height="400" width="700" ></center>
</div>

```py
g = {'Frankfurt': {'Mannheim':85, 'Wurzburg':217, 'Kassel':173},
     'Mannheim': {'Frankfurt':85, 'Karlsruhe':80},
     'Karlsruhe': {'Augsburg':250, 'Mannheim':80},
     'Augsburg': {'Karlsruhe':250, 'Munchen':84},
     'Wurzburg': {'Erfurt':186, 'Numberg':103,'Frankfurt':217},
     'Erfurt': {'Wurzburg':186},
     'Numberg': {'Wurzburg':103, 'Stuttgart':183,'Munchen':167},
     'Munchen': {'Numberg':167, 'Augsburg':84,'Kassel':502},
     'Kassel': {'Frankfurt':173, 'Munchen':502},
     'Stuttgart': {'Numberg':183}
     }
graph = Graph(g)
print("Vertices of graph:")
print(graph.vertices())
print("Edges of graph:")
print(graph.edges())
```


<pre style="font-family:courier new,monospace; background-color:#f6c6529c; color:#000000">
Vertices of graph:
['Mannheim', 'Erfurt', 'Munchen', 'Numberg', 'Stuttgart', 'Augsburg', 'Kassel', 'Frankfurt', 'Wurzburg', 'Karlsruhe']
Edges of graph:
[['Mannheim', 'Frankfurt', 85], ['Mannheim', 'Karlsruhe', 80], ['Erfurt', 'Wurzburg', 186], ['Munchen', 'Numberg', 167], ['Munchen', 'Augsburg', 84], ['Munchen', 'Kassel', 502], ['Numberg', 'Stuttgart', 183], ['Numberg', 'Wurzburg', 103], ['Numberg', 'Munchen', 167], ['Stuttgart', 'Numberg', 183], ['Augsburg', 'Munchen', 84], ['Augsburg', 'Karlsruhe', 250], ['Kassel', 'Munchen', 502], ['Kassel', 'Frankfurt', 173], ['Frankfurt', 'Mannheim', 85], ['Frankfurt', 'Wurzburg', 217], ['Frankfurt', 'Kassel', 173], ['Wurzburg', 'Numberg', 103], ['Wurzburg', 'Erfurt', 186], ['Wurzburg', 'Frankfurt', 217], ['Karlsruhe', 'Mannheim', 80], ['Karlsruhe', 'Augsburg', 250]]
</pre>

Lets say we are given a graph with the cities of Germany and respective distance between them. **You want to find out how to go from Frankfurt (The starting node) to Munchen**. There might be many ways in which you can traverse the graph but you need to find how many cities you will need to visit on a minimum to go from frankfurt to Munchen)
This problem is analogous to finding out distance between nodes in an unweighted graph.

The algorithm that we use here is called as **Breadth First Search**.

```py
def min_num_edges_between_nodes(graph,start_node):
    distance = 0
    shortest_path = []
    queue = [start_node] #FIFO
    levels = {}
    levels[start_node] = 0
    shortest_paths = {}
    shortest_paths[start_node] = ":"
    visited = [start_node]
    while len(queue)!=0:
        start = queue.pop(0)
        neighbours = graph[start]
        for neighbour,_ in neighbours.iteritems():
            if neighbour not in visited:
                queue.append(neighbour)
                visited.append(neighbour)
                levels[neighbour] = levels[start]+1
                shortest_paths[neighbour] = shortest_paths[start] +"->"+ start
    return levels, shortest_paths
```

What we do in the above piece of code is create a queue and traverse it based on levels.
We start with Frankfurt as starting node.
We loop through its neighbouring cities(Menheim, Wurzburg and Kassel) and push them into the queue.
We keep track of what level they are at and also the path through which we reached them.
Since we are popping a first element of a queue we are sure we will visit cities in the order of their level.

Checkout this good [post](https://medium.com/basecs/breaking-down-breadth-first-search-cebe696709d9) about BFS to understand more about queues and BFS.

```py
min_num_edges_between_nodes(g,'Frankfurt')
```

<pre style="font-family:courier new,monospace; background-color:#f6c6529c; color:#000000">
  ({'Augsburg': 3,
  'Erfurt': 2,
  'Frankfurt': 0,
  'Karlsruhe': 2,
  'Kassel': 1,
  'Mannheim': 1,
  'Munchen': 2,
  'Numberg': 2,
  'Stuttgart': 3,
  'Wurzburg': 1},
 {'Augsburg': ':->Frankfurt->Mannheim->Karlsruhe',
  'Erfurt': ':->Frankfurt->Wurzburg',
  'Frankfurt': ':',
  'Karlsruhe': ':->Frankfurt->Mannheim',
  'Kassel': ':->Frankfurt',
  'Mannheim': ':->Frankfurt',
  'Munchen': ':->Frankfurt->Kassel',
  'Numberg': ':->Frankfurt->Wurzburg',
  'Stuttgart': ':->Frankfurt->Wurzburg->Numberg',
  'Wurzburg': ':->Frankfurt'})
</pre>

I did this example to show how  BFS algorithm works.
We can extend this algorithm to find out connected components in an unconnected graph.
Lets say we need to find groups of unconnected vertices in the graph.

For example: the below graph has 3 unconnected sub-graphs. Can we find what nodes belong to a particular subgraph?

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="https://upload.wikimedia.org/wikipedia/commons/8/85/Pseudoforest.svg"  height="400" width="700" ></center>
</div>

```py
#We add another countries in the loop
graph = Graph(g)
graph.add_edge(("Mumbai", "Delhi"),400)
graph.add_edge(("Delhi", "Kolkata"),500)
graph.add_edge(("Kolkata", "Bangalore"),600)
graph.add_edge(("TX", "NY"),1200)
graph.add_edge(("ALB", "NY"),800)

g = graph.adj_mat()

def bfs_connected_components(graph):
    connected_components = []
    nodes = graph.keys()

    while len(nodes)!=0:
        start_node = nodes.pop()
        queue = [start_node] #FIFO
        visited = [start_node]
        while len(queue)!=0:
            start = queue[0]
            queue.remove(start)
            neighbours = graph[start]
            for neighbour,_ in neighbours.iteritems():
                if neighbour not in visited:
                    queue.append(neighbour)
                    visited.append(neighbour)
                    nodes.remove(neighbour)
        connected_components.append(visited)

    return connected_components

print bfs_connected_components(g)
```

The above code is similar to the previous BFS code. We keep all the vertices of the graph in the nodes list. We take a node from the nodes list and start BFS on it. as we visit a node we remove that node from the nodes list. Whenever the BFS completes we start again with another node in the nodes list until the nodes list is empty.

<pre style="font-family:courier new,monospace; background-color:#f6c6529c; color:#000000">[['Kassel',
  'Munchen',
  'Frankfurt',
  'Numberg',
  'Augsburg',
  'Mannheim',
  'Wurzburg',
  'Stuttgart',
  'Karlsruhe',
  'Erfurt'],
 ['Bangalore', 'Kolkata', 'Delhi', 'Mumbai'],
 ['NY', 'ALB', 'TX']]
</pre>

As you can see we are able to find distinct components in our data. Just by using Edges and Vertices. This algorithm could be run on different data to satisfy any use case I presented above.

But Normally using Connected Components for a retail case will involve a lot of data and you will need to scale this algorithm.

## Connected Components in PySpark

Below is an implementation from this paper on [Connected Components in
MapReduce and Beyond](https://ai.google/research/pubs/pub43122) from Google Research. Read the PPT to understand the implementation better.
Some ready to use code for you.

```py
def create_edges(line):
    a = [int(x) for x in line.split(" ")]
    edges_list=[]

    for i in range(0, len(a)-1):
        for j in range(i+1 ,len(a)):
            edges_list.append((a[i],a[j]))
            edges_list.append((a[j],a[i]))
    return edges_list

# adj_list.txt is a txt file containing adjacency list of the graph.
adjacency_list = sc.textFile("adj_list.txt")

edges_rdd = adjacency_list.flatMap(lambda line : create_edges(line)).distinct()

def largeStarInit(record):
    a, b = record
    yield (a,b)
    yield (b,a)

def largeStar(record):
    a, b = record
    t_list = list(b)
    t_list.append(a)
    list_min = min(t_list)
    for x in b:
        if a < x:
            yield (x,list_min)

def smallStarInit(record):
    a, b = record
    if b<=a:
        yield (a,b)
    else:
        yield (b,a)

def smallStar(record):
    a, b = record
    t_list = list(b)
    t_list.append(a)
    list_min = min(t_list)
    for x in t_list:
        if x!=list_min:
            yield (x,list_min)

#Handle case for single nodes
def single_vertex(line):
    a = [int(x) for x in line.split(" ")]
    edges_list=[]
    if len(a)==1:
        edges_list.append((a[0],a[0]))
    return edges_list

iteration_num =0
while 1==1:
    if iteration_num==0:
        print "iter", iteration_num
        large_star_rdd = edges_rdd.groupByKey().flatMap(lambda x : largeStar(x))
        small_star_rdd = large_star_rdd.flatMap(lambda x : smallStarInit(x)).groupByKey().flatMap(lambda x : smallStar(x)).distinct()
        iteration_num += 1

    else:
        print "iter", iteration_num
        large_star_rdd = small_star_rdd.flatMap(lambda x: largeStarInit(x)).groupByKey().flatMap(lambda x : largeStar(x)).distinct()
        small_star_rdd = large_star_rdd.flatMap(lambda x : smallStarInit(x)).groupByKey().flatMap(lambda x : smallStar(x)).distinct()
        iteration_num += 1
    #check Convergence

    changes = (large_star_rdd.subtract(small_star_rdd).union(small_star_rdd.subtract(large_star_rdd))).collect()
    if len(changes) == 0 :
        break

single_vertex_rdd = adjacency_list.flatMap(lambda line : single_vertex(line)).distinct()

answer = single_vertex_rdd.collect() + large_star_rdd.collect()

print answer[:10]
```


## Or Use GraphFrames in PySpark

To Install graphframes:

I ran on command line: pyspark --packages graphframes:graphframes:0.5.0-spark2.1-s_2.11 which opened up my notebook and installed graphframes after i try to import in my notebook.

The string to be formatted as : graphframes:(latest version)-spark(your spark version)-s_(your scala version).

*Checkout* [this guide on how to use GraphFrames](http://go.databricks.com/hubfs/notebooks/3-GraphFrames-User-Guide-python.html) for more information.

```py
from graphframes import *
def vertices(line):
    vert = [int(x) for x in line.split(" ")]
    return vert

vertices = adjacency_list.flatMap(lambda x: vertices(x)).distinct().collect()
vertices = sqlContext.createDataFrame([[x] for x in vertices], ["id"])

def create_edges(line):
    a = [int(x) for x in line.split(" ")]
    edges_list=[]
    if len(a)==1:
        edges_list.append((a[0],a[0]))
    for i in range(0, len(a)-1):
        for j in range(i+1 ,len(a)):
            edges_list.append((a[i],a[j]))
            edges_list.append((a[j],a[i]))
    return edges_list

edges = adjacency_list.flatMap(lambda x: create_edges(x)).distinct().collect()
edges = sqlContext.createDataFrame(edges, ["src", "dst"])

g = GraphFrame(vertices, edges)
sc.setCheckpointDir(".")

# graphframes uses the same paper we referenced apparently
cc = g.connectedComponents()
print cc.show()
```

The GraphFrames library implements the CC algorithm as well as a variety of other graph algorithms.

The above post was a lot of code but hope it was helpful. It took me a lot of time to implement the algorithm so wanted to make it easy for the folks.

If you want to read up more on Graph Algorithms here is an [Graph Analytics for Big Data course on Coursera by UCSanDiego](https://imp.i384100.net/9LLABy) which I highly recommend to learn the basics of graph theory.

## References

1. [Graphs in Python](https://www.python-course.eu/graphs_python.php)
2. [A Gentle Intoduction to Graph Theory Blog](https://medium.com/basecs/a-gentle-introduction-to-graph-theory-77969829ead8)
3. [Graph Analytics for Big Data course on Coursera by UCSanDiego](https://imp.i384100.net/9LLABy)
