
---
title:   How I cracked my MLE interview at Facebook
date:  2020-10-30
draft: false
url : blog/2020/10/30/mle-fb-interview/
slug: mle-fb-interview

Keywords:
- machine learning
- interview
- job
- facebook


Tags:
- Machine Learning
- Algorithms
- Jobs


Categories:
- Programming
- Learning Resources


description:

thumbnail : /images/mle-fb-interview/main.png
image : /images/mle-fb-interview/main.png
toc : false
type : "post"
---


It was August last year and I was in the process of giving interviews. By that point in time, I was already interviewing for Google India and Amazon India for Machine Learning and Data Science roles respectively. And then my senior advised me to apply for a role in Facebook London.

And so I did. Contacted a recruiter on LinkedIn, who introduced me to another one and my process started after a few days for the role of Machine Learning Engineer.

Now Facebook has a pretty different process when it comes to hiring Machine learning engineers. They do coding rounds, system design, and machine learning design interviews to select future employees. Now as far as my experience as a data scientist was concerned I was pretty okay with the Machine learning design interviews but the other interviews still scared me. I had recently failed a Google interview for Machine Learning Software Engineer in the first round itself just because I was not prepared for Data Structure questions.

Later as I studied for the FB coding interview, I realized that I took it a little light and that I was not prepared for the coding interviews at all.

***In this post, I would outline my approach to all these different interviews and how the whole process went round by round for someone who is interested in MLE roles at big organizations like FB.***

So, once I was connected to the recruiter, the next step was a telephonic interview.

---
## 1. Telephonic Interview:

This was a very basic Data Structure interview and sort of a basic sanity check. I guess FB just wants to give you some more time to prepare for the coming rounds and also see whether it would be worth to call you for the onsite rounds. For me, this interview lasted 45 minutes on a Video Call. The interviewer started by telling me about his profile at Facebook, and in turn asking about my profile for the first 10 minutes or so.

Then, I was given 2 very basic array and dictionary-based problems to solve. The interviewer shared a [coderpad](http://coderpad.io) link on which I had to solve these problems in any language of my choice (Not Pseudocode) bug-free without any code format options. I was asked the Time-based constraints and the Space-based constraints also for these questions as well. The interview progressed with me coming up with a bad running time like O(n³) and the interviewer asking if I can do better and giving hints when needed.

As I am not allowed to share the exact questions, I would just share some comparable in difficulty but not the same public Leetcode questions for you so that you can understand the difficulty level and practice accordingly.

a) [Monotonic Array](https://leetcode.com/problems/monotonic-array/): An array is *monotonic* if it is either monotone increasing or monotone decreasing. Return true if and only if the given array A is monotonic.

b) [Valid Palindrome](https://leetcode.com/problems/valid-palindrome/): Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.

### ***My Plan of Action For this Interview?***

This was only the second-ever Data Structure interview I was giving and I wanted to prepare a little after my dismal performance in the first one. So, I started with just understanding the basics of Data Structures using the [Cracking the Coding Interview](https://amzn.to/3fKwwBh) book by Gayle Laakmann McDowell. The book contains a lot of preparation tips as well and you would be prudent to go and read them. The best thing I like about this book is that it's very concise unlike [Thomas Cormen Introduction to Algorithms](https://amzn.to/3mktF5r) and just gives the right amount of background for coding interviews. Every Data structure is explained very concisely in 2–3 pages, some questions are solved around that particular topic and then a few practice questions are given. This book also restricts itself in a way that it only has the most often asked Data Structures. For Example, AVL Trees and Red-Black Trees are kept in the advanced section and not in the trees and graphs chapter as they are often not asked very often in a time-bound interview setting.

I started my preparation by creating a list of topics I would need to prepare. You could prepare even more topics but these are the bare minimum for these interviews.

**Data Structures:** Array, Sets, Stack/Queue, Hashmap/Dictionary, Tree/Binary Tree, Heap, Graphs.

**Algorithms:** Divide-and-Conquer, DP/memoization, Recursion, Binary Search, BFS/DFS, Tree traversals.

Then I proceeded to read about them using the Cracking the coding interview book and solved a lot of easy-level problems and a few medium-level problems on Leetcode for them. There are other platforms too to practice online but I liked Leetcode due to its good design with no ads and the class-based programming structure to the solutions. It also provided a good way to search for questions on various topics along with difficulty. I also gave many Mock Interviews on Leetcode just to practice. I did this for around a week or two, spending around 3–4 hours every day.

I also started auditing the [Algorithm Specialization on Coursera by UCSanDiego](https://click.linksynergy.com/deeplink?id=lVarvwc5BD0&mid=40328&murl=https%3A%2F%2Fwww.coursera.org%2Fspecializations%2Fdata-structures-algorithms) around this time period, which sort of provided me with an idea about the sort of content taught at undergraduate universities to deal with the coding interviews.

I also wrote some blog posts around what I learned and tried to explain it simply. You are free to check them out at my blog.

* [A simple introduction to Linked Lists for Data Scientists](https://mlwhiz.com/blog/2020/01/28/ll/)

* [3 Programming concepts for Data Scientists](https://mlwhiz.com/blog/2019/12/09/pc/) — Recursion/Memoization, DP and Binary Search

* [Dynamic Programming for Data Scientists](https://mlwhiz.com/blog/2020/01/28/dp/)

* [Handling Trees in Data Science Algorithmic Interview](https://mlwhiz.com/blog/2020/01/29/altr/)

***TLDR; **Just jot down the topics you have to prepare for and practice a lot of easy questions on each topic. And maybe a few medium ones too.*

Once, I was done with the telephonic interviews, the recruiter returned in a short period of 1 day and set up a call to explain the process for the onsite interview. ***The onsite round was to happen in London and I was genuinely excited about the prospect to travel to London. An all-paid trip. ***There were to happen around 5 more rounds which I am going to talk about next. I took an interview date in 2 months so that I get some time to prepare. Anyways the VISA and the whole process took up taking a little bit more than that.

Once in London, I reached the Facebook Office from the hotel they provided at around 9 on D Day. A full one hour before the scheduled time because I was anxious and I normally try to reach interviews on/before time(More so when I am giving them). I knew beforehand the whole itinerary of the day as it was shared with me by my recruiter. I also knew which interview would happen at what time and who would take it. It was in fact the most organized interview experience I have rather had.

---

## 2. Onsite Coding Round 1

![](/images/mle-fb-interview/0.jpg)

I have been in Data Science for so long that I read DS as Data Science and not Data Structures. And this interview was essentially a pain point for me. This is something that I learned in a period of 2 months rather than having my whole experience around it. And here I was to be evaluated on this rather than all my experience and Data Science background. But as told to me by the recruiter they have pretty much-fixed processes and I had to go through these rounds for the MLE position. So I played along.

As for the interview, It started on time and as before the interviewer introduced himself before going into my profile for a very small time and then straight jumping into the interview questions. ***This time the questions were a little more difficult and a lot of time was given to formulating the approach and around Time and Space complexities of the solution.*** I was asked a medium level string problem which I was able to solve quickly, and a medium level Binary Search Problem which took most of my time but finally I was able to solve that. Some comparable Problems(Not the same ones) with their brief from Leetcode:

a) [Complex Number Multiplication](https://leetcode.com/problems/complex-number-multiplication/): Given two strings representing two complex numbers. You need to return a string representing their multiplication

b) [Kth Smallest in sorted Matrix](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/): Given a *n* x *n* matrix where each of the rows and columns is sorted in ascending order, find the kth smallest element in the matrix.

The interviewer also gave me an option to code on my own laptop which I had specifically carried there since the recruiter already told me about the option of coding on whiteboard/laptop. But remember they don’t allow using any code formatting and IDEs. I just had a basic editor to write code with.

### My Plan of Action For the Coding Interview?

Just the same plan for the telephonic one but more extensive Leetcoding. I remember I did Leetcode straight for 30 days for these coding rounds around 3–4 hours a day. I used to solve as many medium level questions I could with very little time spent on Hard level questions.

---

## **3. Onsite Coding Round 2:**

Till this time I was in my Data Structure groove and ready for anything that the interviewer was gonna throw at me. My state of mind being — *“ What’s the worst that could happen?”*. And so I just carried on. The people at Facebook were really nice as they asked for refreshments before and after each interview and took care of not overextending any interview. There was a lot of stock given to the fact that each interview starts at exactly the time it needs to with a cool-off period of 15 mins between interviews.

Again some comparable(Not the same ones) problems in difficulty to practice along with a brief description from Leetcode:

a) [API Based question Time-Based Key-Value Store](https://leetcode.com/problems/time-based-key-value-store/): Create a key-value store class, that supports two operations — set and get.

b) [Merge k-Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/): You are given an array of k linked-lists lists, each linked-list is sorted in ascending order. Merge all the linked-lists into one sorted linked-list and return it.

My goal in this coding interview was to be able to solve both the problems that the interviewer put in a time period of 40 mins. But, this was a difficult interview and I took most of the time of the interview in the second problem which was of a hard level. Though the interviewer gave hints to steer me towards the right data structure and algorithm. In the end, I was able to solve problem 1 fully and most of problem 2.

A tip for interviewees would be to call out all solutions you have along with the time complexity involved and only start writing code once you both agree on a good solution.

Also, what I found out through these two interviews was that it was really helpful to talk to your interviewers and explain your approach while you are working. They sometimes provide hints and sometimes stop you from going to the wrong tangent. Even telling your interviewer where you are stuck would help you as it sends out the signal to the interviewer as to which direction you are thinking in. This also makes the whole interview more collaborative and I think that is one of the qualities that the interviewers are looking for in a person.

![Food at FB](/images/mle-fb-interview/1.png "Food is just awesome at FB. Though, I wasnt able to enjoy much as I was a little anxious")

Till this point, I was a little exhausted with all that whiteboard coding and just the general interview pressure, and since it was around lunchtime, I went to the Facebook cafeteria with an assigned colleague/buddy. This is the part where you can ask about the company and this time is not rated on the interviews so you can be quite open with questions about Facebook life and such. And you can enjoy a wide variety of food in the Facebook Cafeteria.

---
## 4. System Design

![](/images/mle-fb-interview/2.png)

This was another interview which I dreaded. As you can see, I was dreading most of the interviews as it was quite an unnatural interview format for me. In a system design interview, you are expected to create a service end to end on a whiteboard. Some example problems for you to practice would be:

* How would you design Netflix?

* How would you design Youtube?

* How would you design Twitter/Facebook?

While this might look daunting, it is actually really open-ended when you prepare for it. As in there are no wrong answers.

The way I like to go through this sort of interview is:

1. **Design a very basic system** that resembles the platform and has the basic functionality the interviewer has asked for. For most of the platforms, it would involve drawing boxes for server, client and a database on a whiteboard.

1. **Create a list of features** I would like to have in the system. For example, Follow in social networks, or booking a taxi on Uber, or double ticks in Whatsapp when the message is read, or retweet functionality in Twitter, or FB newsfeed, etc. The sky is the limit when it comes to features and since we all have seen the features these platforms provide it should not be that hard to come up with a feature list.

1. **Add features** to it throughout the interview and expand/change on the very basic design. This might involve adding the feature as well as talking about scaling, handling edge cases, talking about the data structures and databases involved, using caching etc.

1. **Continue adding features** and evolving the system till the end by asking the interviewer what feature they would like to add based on a list of features I provide them.

### My Plan of Action For the System Design Interview?

There are pretty good resources on the internet to prepare for this interview but I would like to mention two which I found very useful:

* [System Design Primer by Donne Martin](https://github.com/donnemartin/system-design-primer): This is the resource that anyone who prepares for system design should go through at least once but honestly many times. The most important topics to learn here are performance, scalability, latency, throughput, Availability, Consistency, CDNs, Databases, Cache, Load balancing etc.

* Youtube Videos of various system design on the most popular services: I am talking about the big ones — Netflix/Youtube/WhatsApp/Facebook/Gmail/Amazon etc. You can find a lot of videos on youtube for system designs of all these services. The one YouTuber I would definitely wanna call out would be [Techdummies](https://www.youtube.com/c/TechDummiesNarendraL) whose videos I watched for all these big platforms. And who really explained concepts in the easiest way at least for me.

I spent a week where I just jumped from watching videos to reading the repository by Donne Martin for this interview preparation back and forth and I think that was just the right way. Also, it was fun to understand the terms you would find a lot of engineers using, so it was a good learning experience as well.

***In the end, the most important thing in this interview is that you need to drive the discussion with minimal input from the interviewer.*** Sometimes the interviewer might request for a specific feature and you should implement it but in the end, it is your system and you need to create and add features you want in the most logical way to succeed in this round.

---
## 5. Behavioural

This interview tries to look at how you handle difficult situations. And, you can prepare for this interview by assimilating and organizing all the work experiences you have had in the past, the problems you faced and solutions you devised. You will need to collect all the instances where you diffused an impossible situation or failed to deal with one.

![](/images/mle-fb-interview/3.png)

All that said the best way to steer this interview is to **Be Yourself!**

For me, this interview was just like a discussion with the interviewer. He started by giving his introduction and the work he was doing at FB. He then asked me about the projects I was working on and we had a small discussion about the ML parts of the projects. Then it was a very normal discussion about how I would solve/behave in a situation like “What is a mistake you made in your career you are not proud of?”(Not the exact question I was asked). It is helpful if you sort of recollect all the good and bad experiences and frame small stories while preparing for this interview. But again the main point is being honest and a perfectly normal answer to a question like — “Did you had a disagreement with your coworker?” could be none if you have had zero disagreements. Be very honest about your answers as the interviewers for these behavioural round can easily look through someone if they lie.

As far as what I think, for me, this interview went pretty well.

---
## 6. ML System Design

![](/images/mle-fb-interview/4.png)

This was the interview which played to my strengths and honestly, I didn't prepare much for it. In this interview, I was expected to create a system to solve an ML problem end to end.

In this interview, the interviewer is just assessing your ability to convert a Business Problem to a Machine Learning system. So you might be given a problem statement like develop a system to create a newsfeed using Machine Learning, or create a system to filter out toxic comments or honestly any Machine learning system.

You would then need to design a system end to end while talking about various aspects of data and data collection, EDA, feature engineering, model evaluation, model testing, putting the model in production and finally maintenance and feedback.

A good resource to prepare for this interview is from Facebook itself: [Introducing the Facebook Field Guide to Machine Learning video series](https://research.fb.com/the-facebook-field-guide-to-machine-learning-video-series/), which is the only preparation I did for this interview.

---
## And that was a day!

And what a day it was. After the interview, I just went out roaming around London to Trafalgar Square and just watched people performing various tricks and antics. And then a walk back to my Hotel.

To end, it was just a nice interview experience and something I didn’t expect would go well for me. Data Structures were the primary reason I never used to apply for ML roles at big organizations but when I read about them, I found them pretty doable and something you can learn if you can put some time into them.

So, I am joining Facebook London just a year later now as a Software Engineer(MLE) with my joining extended due to COVID related reasons. Hoping this experience goes well.

### Continue Learning

If you want to read up more on Algorithms and Data Structures in a more structured way, here is an [Algorithm Specialization on Coursera by UCSanDiego](https://click.linksynergy.com/deeplink?id=lVarvwc5BD0&mid=40328&murl=https%3A%2F%2Fwww.coursera.org%2Fspecializations%2Fdata-structures-algorithms). I sort of audited this course while preparing.

Thanks for the read. I am going to be writing more beginner-friendly posts in the future too. Follow me up at [Medium](https://mlwhiz.medium.com/?source=post_page---------------------------) or Subscribe to my [blog](https://mlwhiz.ck.page/a9b8bda70c) to be informed about them.

Also, a small disclaimer — There might be some affiliate links in this post to relevant resources, as sharing knowledge is never a bad idea.
