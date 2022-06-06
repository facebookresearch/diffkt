---
id: 'exponential_distribution'
title: 'Exponential Distribution and Gradient Descent'
slug: '/tutorials/data_science/exponential_distribution'
---
The Exponential Distribution and Gradient Descent notebook models the elapsed time between events, such as the time 
between each ad clicks or video views with an exponential distribution. For any given , 
the distribution gives the likelihood that  time passes between two consecutive events.
The notebook uses tensors to hold the data, and the optimization algorithm to minimize the problem 
is a gradient descent algorithm, which uses **DiffKT** for automatic differentiation.

:::tip Open tutorial in Github
[Exponential Distribution and Gradient Descent](https://github.com/facebookresearch/diffkt/blob/main/tutorials/exponential_distribution_mle_gradient_descent.ipynb)