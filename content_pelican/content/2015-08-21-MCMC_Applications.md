Title: Behold the power of MCMC
date:  2015-08-21 04:43
comments: true
slug: MCMC_Algorithms_Cryptography
Category: Statistics
Tags: Statistics,python
description: The way MCMC Algorithm Can Be Applied to Problems with infinite state sizes. Worked on Cryptography Substitution Cipher and the Knapsack Applications in this blog post.
Keywords: Statistics, Machine Learning, MCMC, Bayesian Learning, Data Science, Monte carlo Markov Chain, Cryptography, Knapsack,metropolis algorithm explained,mcmc explained, metropolis hastings explained

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/mcmc.png"></center>
</div>

Last time I wrote an article on MCMC and how they could be useful. We learned how MCMC chains could be used to simulate from a random variable whose distribution is partially known i.e. we don't know the normalizing constant.

So MCMC Methods may sound interesting to some (for these what follows is a treat) and for those who don't really appreciate MCMC till now, I hope I will be able to pique your interest by the end of this blog post. 

So here goes. This time we will cover some applications of MCMC in various areas of Computer Science using Python. If you feel the problems difficult to follow with, I would advice you to go back and read the [previous post](http://mlwhiz.com/blog/2015/08/19/MCMC_Algorithms_Beta_Distribution/), which tries to explain MCMC Methods. We Will try to solve the following two problems:

1. **Breaking the Code** - This problem has got somewhat of a great pedigree as this method was suggested by Persi Diaconis- The Mathemagician. So Someone comes to you with the below text. This text looks like gibberish but this is a code, Could you decrypyt it?<br><br>
<em>XZ STAVRK HXVR MYAZ OAKZM JKSSO SO MYR OKRR XDP JKSJRK XBMASD SO YAZ TWDHZ  MYR JXMBYNSKF BSVRKTRM NYABY NXZ BXKRTRZZTQ OTWDH SVRK MYR AKSD ERPZMRXP  KWZMTRP  MYR JXTR OXBR SO X QSWDH NSIXD NXZ KXAZRP ORRETQ OKSI MYR JATTSN  XDP X OXADM VSABR AIJRKORBMTQ XKMABWTXMRP MYR NSKPZ  TRM IR ZRR MYR BYATP  XDP PAR  MYR ZWKHRSD YXP ERRD ZAMMADH NAMY YAZ OXBR MWKDRP MSNXKPZ MYR OAKR  HAVADH MYR JXTIZ SO YAZ YXDPZ X NXKI XDP X KWE XTMRKDXMRTQ  XZ MYR QSWDH NSIXD ZJSFR  YR KSZR  XDP XPVXDBADH MS MYR ERP Z YRXP  ZXAP  NAMY ISKR FADPDRZZ MYXD IAHYM YXVR ERRD RGJRBMRP SO YAI</em><br><br>

2. **The Knapsack Problem** - This problem comes from [Introduction to probability](http://www.amazon.com/Introduction-Probability-Chapman-Statistical-Science-ebook/dp/B00MMOJ19I) by Joseph Blitzstein. You should check out his courses ([STAT110](http://projects.iq.harvard.edu/stat110/handouts) And [CS109](http://cm.dce.harvard.edu/2014/01/14328/publicationListing.shtml)) as they are awesome. Also as it turns out Diaconis was the advisor of Joseph. So you have Bilbo a Thief who goes to Smaug's Lair. He finds M treasures. Each treasure has some Weight and some Gold value. But Bilbo cannot really take all of that. He could only carry a certain Maximum Weight. But being a smart hobbit, he wants to Maximize the value of the treasures he takes. Given the values for weights and value of the treasures and the maximum weight that Bilbo could carry, could you find a good solution? This is known as the Knapsack Problem in Computer Science.

## Breaking the Code

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/security.png"></center>
</div>

So we look at the data and form a hypothesis that the data has been scrambled using a Substitution Cipher. We don't know the encryption key, and we would like to know the Decryption Key so that we can decrypt the data and read the code.

To create this example, this data has actually been taken from Oliver Twist. We scrambled the data using a random encryption key, which we forgot after encrypting and we would like to decrypt this encrypted text using MCMC Chains. The real decryption key actually is "ICZNBKXGMPRQTWFDYEOLJVUAHS"

So lets think about this problem for a little bit. The decryption key could be any 26 letter string with all alphabets appearing exactly once. How many string permutations are there like that? That number would come out to be $26! \approx 10^{26}$ permutations. That is a pretty large number. If we go for using a brute force approach we are screwed. 
So what could we do? MCMC Chains come to rescue. 

We will devise a Chain whose states theoritically could be any of these permutations. Then we will:

1. Start by picking up a random current state. 
2. Create a proposal for a new state by swapping two random letters in the current state.
3. Use a Scoring Function which calculates the score of the current state $Score_C$ and the proposed State $Score_P$.
4. If the score of the proposed state is more than current state, Move to Proposed State.
5. Else flip a coin which has a probability of Heads $Score_P/Score_C$. If it comes heads move to proposed State.
6. Repeat from 2nd State.

If we get lucky we may reach a steady state where the chain has the stationary distribution of the needed states and the state that the chain is at could be used as a solution.

So the Question is what is the scoring function that we will want to use. We want to use a scoring function for each state(Decryption key) which assigns a positive score to each decryption key. This score intuitively should be more if the encrypted text looks more like actual english if decrypted using this decryption key.

So how can we quantify such a function. We will check a long text and calculate some statistics. See how many times one alphabet comes after another in a legitimate long text like War and Peace. For example we want to find out how many times does 'BA' appears in the text or how many times 'TH' occurs in the text.

For each pair of characters $\beta_1$ and $\beta_2$ (e.g. $\beta_1$ = T and $\beta_2$ =H), we let $R(\beta_1,\beta_2)$ record the number of times that specific pair(e.g. "TH") appears consecutively in the reference text.

Similarly, for a putative decryption key x, we let $F_x(\beta_1,\beta_2)$ record the number of times that
pair appears when the cipher text is decrypted using the decryption key x.

We then Score a particular decryption key x using:

$$Score(x) = \prod R(\beta_1,\beta_2)^{F_x(\beta_1,\beta_2)}$$
    

This function can be thought of as multiplying, for each consecutive pair of letters in the decrypted
text, the number of times that pair occurred in the reference text.  Intuitively, the score function
is higher when the pair frequencies in the decrypted text most closely match those of the reference
text,  and  the  decryption  key  is  thus  most  likely  to  be  correct.

To make life easier with calculations we will calculate $log(Score(x))$

So lets start working through the problem step by step.


<pre style="font-size:60%">
<code class="python"># AIM: To Decrypt a text using MCMC approach. i.e. find decryption key which we will call cipher from now on.
import string
import math
import random

# This function takes as input a decryption key and creates a dict for key where each letter in the decryption key
# maps to a alphabet For example if the decryption key is "DGHJKL...." this function will create a dict like {D:A,G:B,H:C....} 
def create_cipher_dict(cipher):
    cipher_dict = {}
    alphabet_list = list(string.ascii_uppercase)
    for i in range(len(cipher)):
        cipher_dict[alphabet_list[i]] = cipher[i]
    return cipher_dict

# This function takes a text and applies the cipher/key on the text and returns text.
def apply_cipher_on_text(text,cipher):
    cipher_dict = create_cipher_dict(cipher) 
    text = list(text)
    newtext = ""
    for elem in text:
        if elem.upper() in cipher_dict:
            newtext+=cipher_dict[elem.upper()]
        else:
            newtext+=" "
    return newtext

# This function takes as input a path to a long text and creates scoring_params dict which contains the 
# number of time each pair of alphabet appears together
# Ex. {'AB':234,'TH':2343,'CD':23 ..}
def create_scoring_params_dict(longtext_path):
    scoring_params = {}
    alphabet_list = list(string.ascii_uppercase)
    with open(longtext_path) as fp:
        for line in fp:
            data = list(line.strip())
            for i in range(len(data)-1):
                alpha_i = data[i].upper()
                alpha_j = data[i+1].upper()
                if alpha_i not in alphabet_list and alpha_i != " ":
                    alpha_i = " "
                if alpha_j not in alphabet_list and alpha_j != " ":
                    alpha_j = " "
                key = alpha_i+alpha_j
                if key in scoring_params:
                    scoring_params[key]+=1
                else:
                    scoring_params[key]=1
    return scoring_params

# This function takes as input a text and creates scoring_params dict which contains the 
# number of time each pair of alphabet appears together
# Ex. {'AB':234,'TH':2343,'CD':23 ..}

def score_params_on_cipher(text):
    scoring_params = {}
    alphabet_list = list(string.ascii_uppercase)
    data = list(text.strip())
    for i in range(len(data)-1):
        alpha_i =data[i].upper()
        alpha_j = data[i+1].upper()
        if alpha_i not in alphabet_list and alpha_i != " ":
            alpha_i = " "
        if alpha_j not in alphabet_list and alpha_j != " ":
            alpha_j = " "
        key = alpha_i+alpha_j
        if key in scoring_params:
            scoring_params[key]+=1
        else:
            scoring_params[key]=1
    return scoring_params

# This function takes the text to be decrypted and a cipher to score the cipher.
# This function returns the log(score) metric

def get_cipher_score(text,cipher,scoring_params):
    cipher_dict = create_cipher_dict(cipher)
    decrypted_text = apply_cipher_on_text(text,cipher)
    scored_f = score_params_on_cipher(decrypted_text)
    cipher_score = 0
    for k,v in scored_f.iteritems():
        if k in scoring_params:
            cipher_score += v*math.log(scoring_params[k])
    return cipher_score

# Generate a proposal cipher by swapping letters at two random location
def generate_cipher(cipher):
    pos1 = random.randint(0, len(list(cipher))-1)
    pos2 = random.randint(0, len(list(cipher))-1)
    if pos1 == pos2:
        return generate_cipher(cipher)
    else:
        cipher = list(cipher)
        pos1_alpha = cipher[pos1]
        pos2_alpha = cipher[pos2]
        cipher[pos1] = pos2_alpha
        cipher[pos2] = pos1_alpha
        return "".join(cipher)

# Toss a random coin with robability of head p. If coin comes head return true else false.
def random_coin(p):
    unif = random.uniform(0,1)
    if unif>=p:
        return False
    else:
        return True
    
# Takes as input a text to decrypt and runs a MCMC algorithm for n_iter. Returns the state having maximum score and also
# the last few states 
def MCMC_decrypt(n_iter,cipher_text,scoring_params):
    current_cipher = string.ascii_uppercase # Generate a random cipher to start
    state_keeper = set()
    best_state = ''
    score = 0
    for i in range(n_iter):
        state_keeper.add(current_cipher)
        proposed_cipher = generate_cipher(current_cipher)
        score_current_cipher = get_cipher_score(cipher_text,current_cipher,scoring_params)
        score_proposed_cipher = get_cipher_score(cipher_text,proposed_cipher,scoring_params)
        acceptance_probability = min(1,math.exp(score_proposed_cipher-score_current_cipher))
        if score_current_cipher>score:
            best_state = current_cipher
        if random_coin(acceptance_probability):
            current_cipher = proposed_cipher
        if i%500==0:
            print "iter",i,":",apply_cipher_on_text(cipher_text,current_cipher)[0:99]
    return state_keeper,best_state

## Run the Main Program:

scoring_params = create_scoring_params_dict('war_and_peace.txt')

plain_text = "As Oliver gave this first proof of the free and proper action of his lungs, \
the patchwork coverlet which was carelessly flung over the iron bedstead, rustled; \
the pale face of a young woman was raised feebly from the pillow; and a faint voice imperfectly \
articulated the words, Let me see the child, and die. \
The surgeon had been sitting with his face turned towards the fire: giving the palms of his hands a warm \
and a rub alternately. As the young woman spoke, he rose, and advancing to the bed's head, said, with more kindness \
than might have been expected of him: "

encryption_key = "XEBPROHYAUFTIDSJLKZMWVNGQC"
cipher_text = apply_cipher_on_text(plain_text,encryption_key)
decryption_key = "ICZNBKXGMPRQTWFDYEOLJVUAHS"

print"Text To Decode:", cipher_text
print "\n"
states,best_state = MCMC_decrypt(10000,cipher_text,scoring_params)
print "\n"
print "Decoded Text:",apply_cipher_on_text(cipher_text,best_state)
print "\n"
print "MCMC KEY FOUND:",best_state
print "ACTUAL DECRYPTION KEY:",decryption_key
</code></pre>

<div style="margin-top: 9px; margin-bottom: 10px;">
<center><img src="/images/result1_MCMC.png"></center>
</div>

This chain converges around the 2000th iteration and we are able to unscramble the code. That's awesome!!!
Now as you see the MCMC Key found is not exactly the encryption key. So the solution is not a deterministic one, but we can see that it does not actually decrease any of the value that the MCMC Methods provide. Now Lets Help Bilbo :)

## The Knapsack Problem

Restating, we have Bilbo a Thief who goes to Smaug's Lair. He finds M treasures. Each treasure has some Weight and some Gold value. But Bilbo cannot really take all of that. He could only carry a certain Maximum Weight. But being a smart hobbit, he wants to Maximize the value of the treasures he takes. Given the values for weights and value of the treasures and the maximum weight that Bilbo could carry, could you find a good solution?

So in this problem we have an $1$x$M$ array of Weight Values W, Gold Values G and a value for the maximum weight $w_{MAX}$ that Bilbo can carry.
We want to find out an $1$x$M$ array $X$ of 1's and 0's, which holds weather Bilbo Carries a particular treasure or not. 
This array needs to follow the constraint $WX^T < w_{MAX}$ and we want to maximize $GX^T$ for a particular state X.(Here the T means transpose)

So lets first discuss as to how we will create a proposal from a previous state. 

1. Pick a random index from the state and toggle the index value. 
2. Check if we satisfy our constraint. If yes this state is the proposal state.
3. Else pick up another random index and repeat.

We also need to think about the Scoring Function. 
We need to give high values to states with high gold value. We will use:
<br>

$$Score(X)=e^{\beta GX^T}$$

We give exponentially more value to higher score. The Beta here is a +ve constant. But how to choose it? If $\beta$ is big we will give very high score to good solutions and the chain will not be able to try new solutions as it can get stuck in local optimas. If we give a small value the chain will not converge to very good solutions. So weuse an Optimization Technique called **[Simulated Annealing](https://en.wikipedia.org/wiki/Simulated_annealing)** i.e. we will start with a small value of $\beta$ and increase as no of iterations go up.
That way the chain will explore in the starting stages and stay at the best solution in the later stages.

So now we have everything we need to get started

<pre style="font-size:60%">
<code class="python">import numpy as np

W = [20,40,60,12,34,45,67,33,23,12,34,56,23,56]
G = [120,420,610,112,341,435,657,363,273,812,534,356,223,516]
W_max = 150

# This function takes a state X , The gold vector G and a Beta Value and return the Log of score
def score_state_log(X,G,Beta):
    return Beta*np.dot(X,G)

# This function takes as input a state X and the number of treasures M, The weight vector W and the maximum weight W_max
# and returns a proposal state
def create_proposal(X,W,W_max):
    M = len(W)
    random_index = random.randint(0,M-1)
    #print random_index
    proposal = list(X)
    proposal[random_index] = 1 - proposal[random_index]  #Toggle
    #print proposal
    if np.dot(proposal,W)<=W_max:
        return proposal
    else:
        return create_proposal(X,W,W_max)
    
# Takes as input a text to decrypt and runs a MCMC algorithm for n_iter. Returns the state having maximum score and also
# the last few states 
def MCMC_Golddigger(n_iter,W,G,W_max, Beta_start = 0.05, Beta_increments=.02):
    M = len(W)
    Beta = Beta_start
    current_X = [0]*M # We start with all 0's
    state_keeper = []
    best_state = ''
    score = 0
    
    for i in range(n_iter):
        state_keeper.append(current_X)
        proposed_X = create_proposal(current_X,W,W_max)

        score_current_X = score_state_log(current_X,G,Beta)
        score_proposed_X = score_state_log(proposed_X,G,Beta)
        acceptance_probability = min(1,math.exp(score_proposed_X-score_current_X))
        if score_current_X>score:
            best_state = current_X
        if random_coin(acceptance_probability):
            current_X = proposed_X
        if i%500==0:
            Beta += Beta_increments 
        # You can use these below two lines to tune value of Beta
        #if i%20==0:
        #    print "iter:",i," |Beta=",Beta," |Gold Value=",np.dot(current_X,G)
            
    return state_keeper,best_state
</code></pre>

Running the Main program:

<pre style="font-size:60%">
<code class="python">max_state_value =0 
Solution_MCMC = [0]
for i in range(10):
    state_keeper,best_state = MCMC_Golddigger(50000,W,G,W_max,0.0005, .0005)
    state_value=np.dot(best_state,G)
    if state_value>max_state_value:
        max_state_value = state_value
        Solution_MCMC = best_state

print "MCMC Solution is :" , str(Solution_MCMC) , "with Gold Value:", str(max_state_value)
</code></pre>

<pre style="font-family:courier new,monospace; background-color:#f6c6529c; color:#000000">
MCMC Solution is : [0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0] with Gold Value: 2435
</pre>

Now I won't say that this is the best solution. The deterministic solution using DP will be the best for such use case but sometimes when the problems gets large, having such techniques at disposal becomes invaluable.

So tell me What do you think about MCMC Methods? 

Also, If you find any good applications or would like to apply these techniques to some area, I would really be glad to know about them and help if possible. 

The codes for both examples are sourced at [Github](https://github.com/MLWhiz/MCMC_Project)


## References and Sources:
1. [Introduction to Probability Joseph K Blitzstein, Jessica Hwang](http://www.amazon.com/Introduction-Probability-Chapman-Statistical-Science-ebook/dp/B00MMOJ19I)
2. [Wikipedia](https://en.wikipedia.org/wiki/)
3. [The Markov Chain Monte Carlo Revolution, Persi Diaconis](http://statweb.stanford.edu/~cgates/PERSI/papers/MCMCRev.pdf)
4. [Decrypting Classical Cipher Text Using Markov Chain Monte Carlo, Jian Chen and Jeffrey S. Rosenthal](http://www.utstat.toronto.edu/wordpress/WSFiles/technicalreports/1005.pdf)

One of the newest and best resources that you can keep an eye on is the [Bayesian Methods for Machine Learning](https://www.coursera.org/specializations/aml?siteID=lVarvwc5BD0-AqkGMb7JzoCMW0Np1uLfCA&utm_content=2&utm_medium=partners&utm_source=linkshare&utm_campaign=lVarvwc5BD0) course in the [Advanced machine learning specialization](https://www.coursera.org/specializations/aml?siteID=lVarvwc5BD0-AqkGMb7JzoCMW0Np1uLfCA&utm_content=2&utm_medium=partners&utm_source=linkshare&utm_campaign=lVarvwc5BD0) created jointly by Kazanova(Number 3 Kaggler at the time of writing)

Apart from that I also found a course on **[Bayesian Statistics on Coursera](https://click.linksynergy.com/link?id=lVarvwc5BD0&offerid=495576.8910375858&type=2&murl=https%3A%2F%2Fwww.coursera.org%2Flearn%2Fbayesian)**. In the process of doing it right now so couldn't really comment on it. But since I had done an course on **[Inferential Statistics](https://click.linksynergy.com/link?id=lVarvwc5BD0&offerid=495576.8839843074&type=2&murl=https%3A%2F%2Fwww.coursera.org%2Flearn%2Finferential-statistics-intro)** taught by the same professor before(Mine Çetinkaya-Rundel), I am very hopeful for this course. Let's see.

Also look out for these two books to learn more about MCMC. I have not yet read them whole but still I liked whatever I read:

<div style="margin-left:1em ; text-align: center;">

<a target="_blank"  href="https://www.amazon.com/gp/product/1439840954/ref=as_li_tl?ie=UTF8&camp=1789&creative=9325&creativeASIN=1439840954&linkCode=as2&tag=mlwhizcon-20&linkId=d55979088adc0aabeaed88f4f14b48b6"><img border="0" src="//ws-na.amazon-adsystem.com/widgets/q?_encoding=UTF8&MarketPlace=US&ASIN=1439840954&ServiceVersion=20070822&ID=AsinImage&WS=1&Format=_SL250_&tag=mlwhizcon-20" ></a><img src="//ir-na.amazon-adsystem.com/e/ir?t=mlwhizcon-20&l=am2&o=1&a=1439840954" width="1" height="1" border="0" alt="" style="border:none !important; margin:0px !important;" />
</t></t>
<a target="_blank"  href="https://www.amazon.com/gp/product/1584885874/ref=as_li_tl?ie=UTF8&camp=1789&creative=9325&creativeASIN=1584885874&linkCode=as2&tag=mlwhizcon-20&linkId=ee3e2a0bc99359d6c5db0463ab1abb13"><img border="0" src="//ws-na.amazon-adsystem.com/widgets/q?_encoding=UTF8&MarketPlace=US&ASIN=1584885874&ServiceVersion=20070822&ID=AsinImage&WS=1&Format=_SL250_&tag=mlwhizcon-20" ></a><img src="//ir-na.amazon-adsystem.com/e/ir?t=mlwhizcon-20&l=am2&o=1&a=1584885874" width="1" height="1" border="0" alt="" style="border:none !important; margin:0px !important;" />
</div>

Both these books are pretty high level and hard on math. But these are the best texts out there too. :)
