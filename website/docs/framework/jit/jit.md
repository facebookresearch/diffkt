---
id: 'jit'
title: 'Just In Time Optimization'
slug: '/framework/jit'
---
:::tip Open tutorial in Github
[Mass-Spring System with Just In Time Optimization](https://github.com/facebookresearch/diffkt/blob/main/tutorials/mass_spring_jit.ipynb)
:::

The Just In Time (jit) optimization api produces a optimized version of **DiffKt** code. This is useful 
for when you repeatedly call a function. On the first call to a jitted function, an optimized 
version is created. On subsequent calls, the optimized version is called, which should result 
in a speed up of the program.

## Jit Tips and Tricks

There are lots of subtle things you need to get right to take full advantage of the jit:

Make sure there is a good `equals()` and `hashCode()` function for the jitted 
function's input type. The jit cache needs that.

For the purposes of the jit, wrapping more of the input is better. For example, if you 
have some inputs that are not active variables of differentiation inside the body of 
the jitted function, it is still valuable to wrap them for the purposes of the jit so that 
you will get a cache hit when the values change. That means you may want to use a 
different (explicit) wrapInput lambda when taking the derivative.

Don't use mutable variables from an enclosing scope. If they are var variables 
(i.e. they don't change) that is OK, but if the value might change from call to call 
of the jitted function, they should be explicit inputs to the function.

Don't have side-effects in the jitted function; it should be a pure function.
That means no print statements, random number generation, or taking the time of day.