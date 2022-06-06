---
id: 'neural_network'
title: 'Neural Networks and Stochastic Gradient Descent'
slug: '/tutorials/data_science/neural_network'
---
In this notebook we will learn how to build a neural network using backpropagation 
and **DiffKt**. Neural networks are not exactly simple, but they are composed of simple mathematical 
techniques working in orchestration. However the calculus behind neural networks can be tedious, 
as derivatives for each layer need to be calculated for gradient descent purposes. Because weights 
and biases are applied in nested functions from each layer, it's mathematically like pulling apart 
an onion layer-by-layer. Thankfully **DiffKt** can take care of this task of calculating gradients for 
weight and bias layers, and leave out the messiness of solving derivatives by hand.


:::tip Open tutorial in Github
[Neural Networks and Stochastic Gradient Descent](https://github.com/facebookresearch/diffkt/blob/main/tutorials/neural_network_sgd.ipynb)