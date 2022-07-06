---
id: 'user_defined_types'
title: 'User Defined Types'
slug: '/framework/user_defined_types'
---
:::tip Open tutorial in Github
[User Defined Types](https://github.com/facebookresearch/diffkt/blob/main/tutorials/user_defined_types.ipynb)
:::

When you create a __[DTensor](http://www.diffkt.org/api/api/org.diffkt/-d-tensor/index.html)__ or __[DScalar](http://www.diffkt.org/api/api/org.diffkt/-d-scalar/index.html)__ variable, internally it has an implementation of a function call `wrap()`, which is invoked during differentiation operations. The internal representation is used for both the calculation of the user defined function and the calculation of its derivative. Alternatively, one can create their own user defined type. A user defined type could be a class with __[DTensor](http://www.diffkt.org/api/api/org.diffkt/-d-tensor/index.html)__ or __[DScalar](http://www.diffkt.org/api/api/org.diffkt/-d-scalar/index.html)__ variables, or a list of __[DTensor](http://www.diffkt.org/api/api/org.diffkt/-d-tensor/index.html)__ or __[DScalar](http://www.diffkt.org/api/api/org.diffkt/-d-scalar/index.html)__ variables, or even more complex types. When defining a user created type, one has to implement the `wrap()` function as part of the type. There are a couple ways to implement the `wrap()` function and have it called, which are discussed below.

The advantage of the user defined type is that one has named-member access of a class instead of placing all the variables in an array or tensor and having to use indexing to access the variables.

The purpose of __[primalAndForwardDerivative()](http://www.diffkt.org/api/api/org.diffkt/primal-and-forward-derivative.html)__ and __[primalAndReverseDerivative()](http://www.diffkt.org/api/api/org.diffkt/primal-and-reverse-derivative.html)__ is to calculate the derivatives of user defined types. The functions take a user defined input type, a user defined output type, and a user defined derivative type. In addition, the user defines a function for the calculations, and possibly a function to extract the derivatives from the calculations and place the results into the user defined derivative type. Also, the lambdas `wrapInput` and `wrapOutput` might need to be defined to get the `wrap()` function called internally in the code. Notice the similarity in names to __[primalAndForwardDerivative()](http://www.diffkt.org/api/api/org.diffkt/primal-and-forward-derivative.html)__ and __[primalAndReverseDerivative()](http://www.diffkt.org/api/api/org.diffkt/primal-and-reverse-derivative.html)__, as an "s" has been added to the end of the function names __[primalAndForwardDerivative()](http://www.diffkt.org/api/api/org.diffkt/primal-and-forward-derivatives.html)__ and __[primalAndReverseDerivative()](http://www.diffkt.org/api/api/org.diffkt/primal-and-reverse-derivative.html)__ .

__[primalAndForwardDerivative()](http://www.diffkt.org/api/api/org.diffkt/primal-and-forward-derivative.html)__ and __[primalAndReverseDerivative()](http://www.diffkt.org/api/api/org.diffkt/primal-and-reverse-derivative.html)__ have essentially the same function signature . The function signatures are:

`fun <Input : Any, Output : Any, Derivative : Any>`__[primalAndForwardDerivative](http://www.diffkt.org/api/api/org.diffkt/primal-and-forward-derivative.html)__`(
x: Input,
f: (Input) -> Output,
wrapInput: ((Input, Wrapper) -> Input)? = null,
wrapOutput: ((Output, Wrapper) -> Output)? = null,
extractDerivative: (Input, Output, (input: DTensor, output: DTensor) -> DTensor) -> Derivative,
): Pair<Output, Derivative>`

and

`fun <Input : Any, Output : Any, Derivative : Any>`__[primalAndReverseDerivative](http://www.diffkt.org/api/api/org.diffkt/primal-and-reverse-derivative.html)__`(
x: Input,
f: (Input) -> Output,
wrapInput: ((Input, Wrapper) -> Input)? = null,
wrapOutput: ((Output, Wrapper) -> Output)? = null,
extractDerivative: (Input, Output, (input: DTensor, output: DTensor) -> DTensor) -> Derivative,
): Pair<Output, Derivative>`

The type for `Input`, `Output`, and `Derivative` are user defined. The user defined types could be a class with __[DScalar](http://www.diffkt.org/api/api/org.diffkt/-d-scalar/index.html)__ or __[DTensor](http://www.diffkt.org/api/api/org.diffkt/-d-tensor/index.html)__ variables, a list with __[DScalar](http://www.diffkt.org/api/api/org.diffkt/-d-scalar/index.html)__ or __[DTensor](http://www.diffkt.org/api/api/org.diffkt/-d-tensor/index.html)__ elements, or something more complex.

The function `f: (Input) -> Output` has to know how to access the variables in the `Input` type and produce a return of `Output` type.

The `Derivative` type has to define all the possible derivates that can be produced from taking the derivative of `f()` with respect to the `Input` type.

The `Input` or `Output` types can inherit the `Differentiable<T>` interface, which knows how to call the `wrap()` function.. If the `Input` and `Output` types do not inherit from the `Differentiable<T>` interface, then a lambda expression needs to written for the `wrapInput` and/or the `wrapOutput` functions to call `wrap()` for the `Input` or `Output` type.