---
id: 'tensors'
title: 'Tensors'
slug: '/framework/tensors'
---
:::tip Open tutorial in Github
[Introduction to Basic API Operations for DiffKt](https://github.com/facebookresearch/diffkt/blob/main/tutorials/intro_to_differentiable_programming.ipynb)
:::

In **DiffKt** there are many different types of differentiable tensors. 
Tensor means a multi-dimensional array. A float scalar is  a 0D tensor. A vector is a 1D tensor.
A 2D array is a 2D tensor. A 3D array is a 3D tensor, and so on.

__[DTensor](http://www.diffkt.org/api/api/org.diffkt/-d-tensor/)__ is the interface for all 
differentiable tensors in **DiffKt**. A differentiable tensor can be a scalar, a 1D tensor, a 2D tensor, 
a 3D tensor, or have even more dimensions. 
Scalars also inherit from __[DTensor]( http://www.diffkt.org/api/api/org.diffkt/-d-tensor/)__. A tensor has a number 
of properties, functions, or extensions defined in the interface. Properties we will discuss 
about __[DTensor]( http://www.diffkt.org/api/api/org.diffkt/-d-tensor)__ are size, rank, shape, isScalar, and indexing.

A tensor has a __[size](http://www.diffkt.org/api/api/org.diffkt/-d-tensor/size.html)__, 
which is the number of elements in the tensor,

A tensor has a __[rank](http://www.diffkt.org/api/api/org.diffkt/-d-tensor/rank.html)__, 
which indicates the number of dimensions: rank 0 - scalar, rank 1 - 1D tensor, rank 2 - 2D tensor, 
rank 3 - 3D tensor, and so on.

A tensor has a __[shape](http://www.diffkt.org/api/api/org.diffkt/-d-tensor/shape.html)__, 
which indicates the number of axes and the length of each axis of the tensor.

A tensor has an boolean property to see if it is a scalar, 
__[isScalar](http://www.diffkt.org/api/api/org.diffkt/-d-tensor/is-scalar.html)__.

Retrieve an element of a tensor use indexing, with the indices indicating the location of the element, 
such as `[0,0]` to get the first element of the 2D array.

__[FloatTensor](http://www.diffkt.org/api/api/org.diffkt/-float-tensor/index.html)__ 
is an an abstract class for the implementation of __[DTensor](http://www.diffkt.org/api/api/org.diffkt/-d-tensor/index.html)__ for floating point numbers. There are multiple types of implementations such as scalar, dense, and sparse tensors.

__[DScalar](http://www.diffkt.org/api/api/org.diffkt/-d-scalar/index.html)__ 
is the interface for all differentiable scalars.

__[FloatScalar](http://www.diffkt.org/api/api/org.diffkt/-float-scalar/index.html)__ 
is an implementation of the interfaces __[DScalar](http://www.diffkt.org/api/api/org.diffkt/-d-scalar/index.html)__ 
and  __[FloatTensor](http://www.diffkt.org/api/api/org.diffkt/-float-tensor/index.html)__.

__[tensorOf](http://www.diffkt.org/api/api/org.diffkt/tensor-of.html)__ is a factory function that 
creates a FloatTensor from a set of float numbers. The initial tensor is a 1D array. After creating a tensor 
with __[tensorOf](http://www.diffkt.org/api/api/org.diffkt/tensor-of.html)__, 
you may need to __[reshape](http://www.diffkt.org/api/api/org.diffkt/reshape.html)__ the tensor to the shape you want.

## Tensor Operations

The __[DTensor](http://www.diffkt.org/api/api/org.diffkt/-d-tensor/index.html)__ interface has many 
operations that can be applied to a tensor. Click on the **Extentions** tab in the Kotlin docs of 
__[DTensor](http://www.diffkt.org/api/api/org.diffkt/-d-tensor/index.html)__ to see all the operations. 
Some of the operations allow the use of traditional arithmatic notation, or operator overloading. We will look at a few of the operations in the below examples:

__['+'](http://www.diffkt.org/api/api/org.diffkt/plus.html)__ or __[plus](http://www.diffkt.org/api/api/org.diffkt/plus.html)__,

__['-'](http://www.diffkt.org/api/api/org.diffkt/minus.html)__ or __[minus](http://www.diffkt.org/api/api/org.diffkt/minus.html)__,

__['*'](http://www.diffkt.org/api/api/org.diffkt/times.html)__ or __[times](http://www.diffkt.org/api/api/org.diffkt/times.html)__,

__['/'](http://www.diffkt.org/api/api/org.diffkt/div.html)__ or __[div](http://www.diffkt.org/api/api/org.diffkt/div.html)__,

__[pow](http://www.diffkt.org/api/api/org.diffkt/pow.html)__,

__[sin](http://www.diffkt.org/api/api/org.diffkt/sin.html)__,

__[cos](http://www.diffkt.org/api/api/org.diffkt/cos.html)__,

__[matmul](http://www.diffkt.org/api/api/org.diffkt/matmul.html)__,

__[sum](http://www.diffkt.org/api/api/org.diffkt/sum.html)__ and,

__[innerProduct](http://www.diffkt.org/api/api/org.diffkt/inner-product.html)__

## Calculating the Derivative of a Scalar Function

There are two different algorithms for calculating the derivative of a 
function over a __[DScalar](http://www.diffkt.org/api/api/org.diffkt/-d-scalar/index.html)__ variable, 
the forward derivative algorithm and the reverse derivative algorithm. The forward derivative algorithm 
is more efficient for when a function has more output variables than input variables. The reverse derivative 
algorithm is more efficient for a function that has more input variables that output variables. For most situations 
of optimizing a scalar function, where the output of the function is a single variable, the reverse derivative 
algorithm is more efficient.

In calling the below functions, one passes a scalar variable, 
a __[DScalar](http://www.diffkt.org/api/api/org.diffkt/-d-scalar/index.html)__, 
to be differentiated and a lambda of the function of the variable. In Kotlin, if you 
declare the function `fun f(x)` then the lambda is `::f`.

__[forwardDerivative](http://www.diffkt.org/api/api/org.diffkt/forward-derivative.html)__ calculates 
the derivative of a function over a __[DScalar](http://www.diffkt.org/api/api/org.diffkt/-d-scalar/index.html)__ 
evaluated at the __[DScalar](http://www.diffkt.org/api/api/org.diffkt/-d-scalar/index.html)__ `x` using 
the forward derivative algorithm.

__[reverseDerivative](http://www.diffkt.org/api/api/org.diffkt/reverse-derivative.html)__ calculates the 
derivative of a function over a __[DScalar](http://www.diffkt.org/api/api/org.diffkt/-d-scalar/index.html)__ 
evaluated at the __[DScalar](http://www.diffkt.org/api/api/org.diffkt/-d-scalar/index.html)__ `x` using 
the reverse derivative algorithm.

In many cases it is more efficient to calculate the orignal scalar function and its derivative at the same time. 
In the below functions, they return a `Pair<DTensor, DTensor>` where the first value is called the `primal`, 
which is the value of a function evaluated at `x`, where `x` is a tensor, and the second value is called 
the `tangent`, which is the derivative of a function evaluated at `x`, where `x` is a tensor.

__[primalAndForwardDerivative](http://www.diffkt.org/api/api/org.diffkt/primal-and-forward-derivative.html)__ 
calculates a function over __[DScalar](http://www.diffkt.org/api/api/org.diffkt/-d-scalar/index.html)__ and its 
derivative evaluated at the __[DScalar](http://www.diffkt.org/api/api/org.diffkt/-d-scalar/index.html)__ `x` using 
the forward derivative algorithm.

__[primalAndReverseDerivative](http://www.diffkt.org/api/api/org.diffkt/primal-and-reverse-derivative.html)__ 
calculates a function over a __[DScalar](http://www.diffkt.org/api/api/org.diffkt/-d-scalar/index.html)__ and its 
derivative evaluated at the __[DScalar](http://www.diffkt.org/api/api/org.diffkt/-d-scalar/index.html)__ `x` using 
the reverse derivative algorithm.

## Derivatives of a Function over a Tensor

The symbol nabla, $\nabla$, is an inverted greek symbol $\Delta$. The gradient of a function over a vector of 
variables is $\nabla f(\mathbf x)$, and is the partial derivatives of the function with respect to each variable. 
The Jacobian of a vector valued function, either $J(\mathbf f(\mathbf x))$ or $\mathbf \nabla \mathbf f( \mathbf x)$ 
is the gradient of each vector component of the function, or the partial derivatives of each vector component of 
the function with respect to each variable.

The partial derivatives of a function with N inputs and 1 output at a point $\mathbf x$, where $\mathbf x$ is a 
vector of size N, or a function $f(\mathbf x):R^N \rightarrow R^1$, is the gradient of the function, 
which is a function $\nabla f(\mathbf x): R^N \rightarrow R^N$. The gradient of a function of N variables, 
where $\mathbf x = \left [ x_1, x_2, \cdots, x_n \right ]$ is

$\nabla f(\mathbf x) = \left [ \frac {\partial f(\mathbf x)} {\partial x_1}, \frac {\partial f(\mathbf x)} 
{\partial x_2}, \cdots, \frac {\partial f(\mathbf x)} {\partial x_n} \right ]^T$.

For example, if $f(x,y) = 4x^2 + 2y$ then $\nabla f(x, y) = \left [ 8x, 2 \right ]^T$, 
where $\nabla f(x,y) = \left [\frac {\partial f(x,y)} {\partial x}, \frac {\partial f(x,y)} {\partial y} \right ]^T$.


The partial derivatives of a function with N inputs and M outputs at a point 
$\mathbf x$, where $\mathbf x$ is of size N, or a function $\mathbf f(\mathbf x): R^N \rightarrow R^M$, 
is the Jacobian of the function, or $\mathbf \nabla \mathbf f(\mathbf x): R^N \rightarrow R^{NxM}$. 
The point $\mathbf x$ is a vector of variables, $\mathbf x = \left [ x_1, x_2, \cdots, x_n \right ]$. 
The function $\mathbf f(\mathbf x)$ is a vector of functions evaluated at 
$\mathbf x$, $\mathbf f(\mathbf x) = \left [ f_1(\mathbf x), f_2(\mathbf x), \cdots, f_m(\mathbf x) \right ]^T$.

The Jacobian of a function is the partial derivatives of each component function by each variable.

$\mathbf \nabla \mathbf f(\mathbf x) = \begin{bmatrix}\frac{\partial f_1}{\partial x_1}\cdots\frac{\partial f_1}{\partial x_n}\\ \hspace{0.5em}\vdots\hspace{0.3em}\ddots\hspace{0.3em}\vdots\\ \frac{\partial f_m}{\partial x_1}\cdots \frac{\partial f_m}{\partial x_n}\end{bmatrix}$.

For example, if $\mathbf f(x,y) = \left[4x^2 + 2y, 2x + 4y^2 \right]$ then

the Jacobian is $\mathbf \nabla \mathbf f(x,y) = \begin{bmatrix} 8x,\hspace{0.5em} 2\\ \hspace{0.5em} 2, 8y\end{bmatrix}$.

__[forwardDerivative](http://www.diffkt.org/api/api/org.diffkt/forward-derivative.html)__ 
calculates the derivative of a function over a tensor, evaluated at the tensor `x`, using 
the forward derivative algorithm.

__[reverseDerivative](http://www.diffkt.org/api/api/org.diffkt/reverse-derivative.html)__ calculates the 
derivative of a function over a tensor, evaluated at the tensor `x`, using the reverse derivative algorithm. 
The reverse derivative algorithm returns the transpose of the derivative calculation, compared to the forward 
derivative algorithm, when the result is a Jacobian or 2D tensor.

In many cases it is more efficient to calculate the orignal function and its partial derivatives at the same time. 
In the below functions, they return a `Pair<DTensor, DTensor>`. The first value is called the `primal`, which is 
the value of a function evaluated at `x`, where `x` is a tensor. The second value is called the `tangent`, which is 
the derivative of a function evaluated at `x`, where `x` is a tensor.

__[primalAndForwardDerivative](http://www.diffkt.org/api/api/org.diffkt/primal-and-forward-derivative.html)__ 
calculates a function over a tensor `x` and its derivative, evaluated at the tensor `x,` using the 
forward derivative algorithm.

__[primalAndReverseDerivative](http://www.diffkt.org/api/api/org.diffkt/primal-and-reverse-derivative.html)__ calculates 
a function over a tensor `x` and its derivative, evaluated at the tensor `x`, using the reverse derivative algorithm. 
The reverse derivative algorithm returns the transpose of the derivative calculation, compared to the forward 
derivative algorithms, when the result is a Jacobian or 2D tensor.