+++
date = "2020-04-01T10:15:23-04:00"
draft = false
title = "Code Gradient Descent From Scratch"

+++

*How to program gradient descent from scratch in python. For that time you got asked in an interview and fumbled.*

<img src="/img/grad_desc/cover.jpeg" alt="grad_desc" width="500" height="500"/>

## Motivation

This is it. You've networked your way through the door by sending approximately 10 LinkedIn messages to perfect strangers and charming the recruiter through that 30 minute phone call summarizing your entire adult professional life. They've sent you...dun dun dun....the assignment. 4 hours they say. Just 4 short hours. "4 hours" you think to yourself "piece of cake". The email comes along with the link to a google doc of instructions. "Just 2 prompts" you think again, "No problem at all. Bet I'll have time to spare." You open the assignment. There. Sprawled across the first google sheet reads "Please minimize this unknown function". Your brain goes dark and you contemplate whether you could pass a fifth grade math test.

## The Prompt

Alright so let's take a fresh look at what this interviewer is trying to get you to do: 

**Given:** You pull the function from something like a pickle file, so you only know the inputs (which you specify) and tbe output (given by the function). 

**Problem:** Find the inputs which minimize the output of this function by computing the gradient at each given point. For example we'll use f(x1,x2) = y 

## Background

This is gradient descent. It's just broken down to the components. You know it. You've learned it before. There are so many articles on gradient descent that I can't justify why writing about it extensively here would provide value. A few good resources are this [article](https://towardsdatascience.com/understanding-the-mathematics-behind-gradient-descent-dde5dc9be06e) for quick review and the classic Andrew Ng [course](https://www.coursera.org/lecture/machine-learning/gradient-descent-8SpIM) for everything you've ever wanted to know. For now I'll remind you of the basics required to complete this challenge. 

1. What is a gradient? <br/>
	
	The book definition: a gradient is the magnitude and direction of change from one point to another.
	
	The practical definition: Before I switched to data science I studied engineering, so my best examples of gradients are rooted in mechanics and fluid dynamics. Hopefully this analogy will be absolutely perfect for one person, and for everyone else I hope it's simple enough. Imagine a lake. Pretty calm typically. <br/>
	
	<img src="/img/grad_desc/calm_lake.jpg" alt="grad_desc" width="500" height="500"/>
	
	Ok now it's a windy day and the water picks up a bit of current.
	
	<img src="/img/grad_desc/windy_lake.jpg" alt="grad_desc" width="500" height="500"/>
	
	When it's windy out does the lake have the same current throughout or does the wind tend to pick up and die down over time? I'd say the latter. So assuming that, the water can't have the same speed and direction across the entire surface right? How could we represent the speed and direction of the water on the surface at different parts of this lake? Well mechanical physics would tell you to use a velocity vector. Which represents the speed and direction. Like so:
	
	![grad_desc](/img/grad_desc/vel_vec.png)
	
	Now this represents one gradient. This is the change in distance (in a particular direction) from one point to another with respect to time. Start combining these gradients together and we get something that looks like this:
	
	![grad_desc](/img/grad_desc/vel_grad.png)
	
	The math definition: A gradient is the slope of the line tangent to the curve at whatever point you are assessing. This means a gradient is the sum of the partial derivative components reprepresting that function at that point. Which brings us to something like this:
	
	<img src="/img/grad_desc/grad_eq.png" alt="grad_desc" width="300" height="100"/>

2. What are we descending and why?

	Now imagine the function above is unknown. Well, if the gradient is the magnitude and direction of change from one point to another then how would we find (perhaps) the minimum point of a function? Follow the gradients. Let's specifically follow the largest gradients that bring us in the right direction of that point. We'll specify a rate that we want to check these gradients (step size) as well as the point we want to start at (x,y,z,etc). Then we'll check each gradient around that point. 
	
	Why do we do this? In machine learning you are typically trying to find some relationship between a predictor and a series of features. In order to converge you optimize a [cost function](https://towardsdatascience.com/coding-deep-learning-for-beginners-linear-regression-part-2-cost-function-49545303d29f) for the lowest amount of error. The way you optimize this is by taking steps in the direction of the lowest error and returning the gradient vector that corresponds. Once you reach a threshold minimum your gradient vector reveals the coefficients which optimize this cost function.
	
	 I would presume that your interviewer, therefore, is testing whether or not you can dive under the hood of your algorithmic vehicle and conduct a bit of manual maintenance. 
	
## Coding the Solution

Alright let's get down to it. I made two functions to solve this problem. The first function will simply compute the maximum gradient given x = (x1, x2). I'm going to presume there are only 2 features (x1,x2) as defined by the prompt. The second function will use that first function to minimize your unknown (given) function f(x1,x2) = y by following gradients in the first function. Starting with the first function we have:

```python
def gradient(func,x):
	# func = the unknown function you are given
	# x = (x1,x2) 
    # Define step size and current y
    step = 1
    y = func(x)
    
    # Check all points around x,y for the largest difference (gradient magnitude)
    # Neg points 
    x_neg = (x[0] - step, x[1]-step)
    y_neg = func(x_neg)
    diff_neg = y - y_neg
    # Pos points
    x_pos = (x[0] + step, x[1]+step)
    y_pos = func(x_pos)
    diff_pos = y - y_pos
    # Pos Neg points 
    x_pn = (x[0]+step, x[1]-step)
    y_pn = func(x_pn)
    diff_pn = y - y_pn
    # Neg Pos points
    x_np = (x[0]-step, x[1]+step)
    y_np = func(x_np)
    diff_np = y - y_np
    diff = [diff_neg, diff_pos, diff_pn, diff_np]
    
    # Compare diffs
    if max(diff) == diff_neg:
        # Returning the magnitude and direction of the gradient
        grad = (x_neg, diff_neg)
    elif max(diff)==diff_pos:
        grad = (x_pos, diff_pos)
    elif max(diff)==diff_pn:
        grad = (x_pn, diff_pn)
    else:
        grad = (x_np, diff_np)
    return grad
```
What's happening here is that I'm grabbing the difference between every point around my given point (x1,x2) and my given point + the step size I instantiated (here it is 1). Then I'll find the minimum of these 4 differences (the maximum negative gradient) and return the direction (x1+step,x2+step) and magnitude (difference) associated. 

Let's take a look at the minimizing function:

```python
def minimize(func, debug=False):
	# func = the unknown function you are given
	# Define where you want to start minimizing from (x)
	# Define the threshold difference to stop at (closest to minimum point)
    x = (600,600)
    diff = 99999999
    stop = 0.00005
    
    # Until you reach threshold (stop) --> minimize
    while abs(diff) > stop:
        grad = gradient(func,x)
        diff = grad[1]
        x = grad[0]
        if debug:
            print(grad[0],grad[1])
    return print("Minimum Point: (%.2f,%.2f)"%(grad[0][0], grad[0][1]))
```
Here I'm essentially just rolling through my gradients using the returned minimum from my gradient function to rerun the loop every time. Once I've reached a stopping difference that is close to zero I return the coordinates (x1,x2) of the function at that point. I'd also recommend visualizing this (if possible) to get a feel for your solution. My example looked like this:

![grad_viz](/img/grad_desc/grad_viz.png)

As you can see there is a local minima in addition to the global minima. I get around this problem by initializing my script at randomly different starting points to ensure I'm reaching the true minimum. 

## In Conclusion

Some form of this problem is pretty typical for a machine learning interview. It shows that you understand the nuts and bolts behind the powerful algorithms you are using. I also think it's a fun puzzle for showing some basic python chops as you really don't need any fancy packages for this. A few references I used to approach this problem are below and I wish you the best of luck in future interviewing endeavors. May you descend gradients with grace.

## References 

- [Wikepedia on Gradients](https://en.wikipedia.org/wiki/Gradient) 
- [Gradient Descent in Python](https://towardsdatascience.com/gradient-descent-in-python-a0d07285742f)
