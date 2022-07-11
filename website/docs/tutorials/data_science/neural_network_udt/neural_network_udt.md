---
id: 'neural_network_udt'
title: 'Neural Network with User-Defined Types'
slug: '/tutorials/data_science/neural_network_udt'
---
In this notebook you will learn how to build a neural network using DiffKt with user-defined types. 
Neural networks are not exactly simple, but they are composed of simple mathematical techniques 
working in orchestration. However the calculus behind neural networks can be tedious, as 
derivatives for each layer need to be calculated for gradient descent purposes. Because weights 
and biases are applied in nested functions from each layer, it's mathematically like pulling apart 
an onion layer-by-layer. Thankfully **DiffKt** can take care of this task of calculating gradients 
for weight and bias layers, and leave out the messiness of solving derivatives by hand. Along the way, 
you will use custom types and demonstrate **DiffKt's** capabilities with its `Wrapper` interface. 

:::tip Open tutorial in Github
[Neural Networks with User-Defined Types](https://github.com/facebookresearch/diffkt/blob/main/tutorials/neural_network_user_defined_types.ipynb)
:::