---
id: 'mle'
title: 'Normal Distribution and Gradient Descent'
slug: '/tutorials/data_science/mle'
---
The Normal Distribution and Gradient Descent notebook shows how to estimate 
the mean and standard deviation of a normal distribution from data using maximal likelihood estimation (MLE). The 
MLE algorithm is implemented using gradient descent, which uses **DiffKt** to calculate the derivatives of the 
mean and standard deviation for use in the algorithm. In this example the data is stored in a tensor and 
the derivatives are calculated from a function over a tensor.

:::tip Open tutorial in Github
[Normal Distribution and Gradient Descent](https://github.com/facebookresearch/diffkt.preopn/blob/main/tutorials/normal_mle_gradient_descent.ipynb)